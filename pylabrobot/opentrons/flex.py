import logging
from contextlib import asynccontextmanager
from typing import (
  Any,
  AsyncIterator,
  Dict,
  Iterable,
  List,
  NamedTuple,
  Optional,
  Set,
  Tuple,
  Type,
  cast,
)

from pylabrobot.opentrons.flex_gripper import FlexGripper
from pylabrobot.opentrons.flex_head import FlexHead1, FlexHead8, FlexHead96, _FlexHead
from pylabrobot.opentrons.flex_wire import ROBOT_AXES, _require_robot_commands, slot_wire_location
from pylabrobot.opentrons.labware_definitions import (
  build_container_definition,
  build_movable_labware_definition,
  build_plate_definition,
  build_tip_rack_definition,
)
from pylabrobot.opentrons.robot import COMMAND_POLL_HEADROOM, OpentronsError, OpentronsRobot
from pylabrobot.opentrons.transport import OpentronsTransport
from pylabrobot.resources import Container, Plate, Resource, TipRack
from pylabrobot.resources.opentrons.flex_deck import FlexDeck
from pylabrobot.resources.trash import Trash

logger = logging.getLogger(__name__)

_OT_NAMESPACE = "opentrons"

# Every definition has a revision 1, so it is the only version safe to assume.
_OT_VERSION = 1

# Discovered pipette channel count -> matching head class.
_CHANNELS_TO_HEAD: Dict[int, Type[_FlexHead]] = {
  1: FlexHead1,
  8: FlexHead8,
  96: FlexHead96,
}


def _corner_offset_z(definition: Dict[str, Any]) -> Optional[float]:
  """How far above its slot a definition puts the labware's origin.

  Schema 2 states it in ``cornerOffsetFromSlot``. Schema 3 dropped the field and
  puts the labware origin on the slot, so the distance is zero -- its ``extents``
  are a bounding box measured from that origin, which is a different quantity and
  not a stand-in for this one. A schema this driver has not been taught answers
  None rather than zero, because reading it as zero would put every well floor
  out by however far the labware really sits above its slot.
  """
  version = definition.get("schemaVersion")
  if version == 2:
    corner = definition.get("cornerOffsetFromSlot")
    if isinstance(corner, dict) and "z" in corner:
      return float(corner["z"])
    return None
  if version == 3:
    return 0.0
  return None


class _LoadedGeometry(NamedTuple):
  """The parts of a run's labware definition a deck z has to be framed against."""

  wells: Dict[str, Dict[str, Any]]
  corner_offset_z: Optional[float]
  """Where the definition puts the labware's origin above its slot. None when the
  definition states it in a form this driver does not read, which refuses a height
  rather than measuring it from a floor assumed to sit at the slot."""
  offset_id: Optional[str]
  """The labware offset the run applied, if any. PyLabRobot creates runs with no
  offsets, so this is None in practice; it is read so a run that does carry one
  fails loudly rather than reporting heights measured against the wrong floor."""


def _has_pipettable_geometry(resource: Resource) -> bool:
  """Whether a real, well-bearing definition can be built from this resource."""
  return isinstance(resource, (Plate, TipRack, Container))


def _not_pipettable_error(resource: Resource) -> OpentronsError:
  """The refusal for a resource no pipettable definition can be built from."""
  return OpentronsError(
    "Cannot build an Opentrons labware definition",
    f"'{resource.name}' ({type(resource).__name__}) has no Opentrons load name, and a "
    "definition can only be built from the geometry of a Plate, TipRack, or Container. "
    "Set resource.ot_load_name to an official Opentrons load name. The gripper can still "
    "move it -- only pipetting needs real well geometry.",
  )


def _warn_grip_distance_discarded(
  name: str, grip_distance_from_top: Optional[float], why: str
) -> None:
  """Log a caller's grip distance being dropped, naming why it cannot apply."""
  if grip_distance_from_top is None:
    return
  logger.warning(
    "grip_distance_from_top=%s ignored for '%s': %s.", grip_distance_from_top, name, why
  )


def _validate_axes(axes: Iterable[str]) -> None:
  """Raise for any axis name the robot does not address."""
  unknown = sorted(set(axes) - ROBOT_AXES)
  if unknown:
    raise ValueError(
      f"unknown axes: {', '.join(unknown)}. Valid axes are: {', '.join(sorted(ROBOT_AXES))}."
    )


def _reported_axis_position(command: Dict[str, Any]) -> Dict[str, float]:
  """The axis positions a completed robot/moveAxes* command reports."""
  return cast(Dict[str, float], command.get("result", {}).get("position", {}))


def _axis_motion_params(
  axis_map: Dict[str, float],
  speed: Optional[float],
  critical_point: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
  """The snake_case payload a robot/moveAxes* command takes (unlike the rest of
  the API, which is camelCase), with every axis name validated first."""
  _validate_axes(axis_map)
  params: Dict[str, Any] = {"axis_map": axis_map}
  if critical_point is not None:
    _validate_axes(critical_point)
    params["critical_point"] = critical_point
  if speed is not None:
    params["speed"] = speed
  return params


class OpentronsFlex(OpentronsRobot):
  """Opentrons Flex liquid handler (plain class, post-#1180 architecture).

  A device shell: it owns the deck, deck-scoped labware loading, and the
  discover-then-compose lifecycle that builds mount-addressed head
  sub-objects (``left``/``right``/``head96``). Liquid-handling ops live on
  the heads, not here — see :mod:`pylabrobot.opentrons.flex_head`.
  """

  def __init__(
    self,
    deck: FlexDeck,
    host: str,
    port: int = 31950,
    transport: Optional[OpentronsTransport] = None,
  ) -> None:
    super().__init__(host=host, port=port, transport=transport)
    self.deck = deck
    self._loaded_labware: Dict[str, str] = {}
    # resource.name -> the geometry the run loaded it with. The run owns this: a
    # labware loaded by an official Opentrons name uses vendor geometry PyLabRobot
    # never sees, so nothing on this side can frame a deck z against it.
    self._loaded_geometry: Dict[str, _LoadedGeometry] = {}
    # resource.name -> (namespace, load_name, version) of an uploaded custom definition.
    self._defined_labware: Dict[str, Tuple[str, str, int]] = {}
    # Names loaded under the non-pipettable movable stub. Tracked separately so
    # a pipetting op refuses them even on a load-cache hit.
    self._stub_labware: Set[str] = set()
    # The labware id the robot last pipetted over, None when unknown. On the robot
    # and not the head: one x/y gantry, so a move by either head moves both.
    self._current_labware_id: Optional[str] = None
    self.left: Optional[_FlexHead] = None
    self.right: Optional[_FlexHead] = None
    self.head96: Optional[_FlexHead] = None
    self.gripper: Optional[FlexGripper] = None
    self._heads: List[_FlexHead] = []

  @asynccontextmanager
  async def _moving_to(self, labware_id: Optional[str]) -> AsyncIterator[None]:
    """Run a gantry move, and take ``labware_id`` as the new pipetting position
    only once the move has landed.

    The tracked position is dropped first because a move that raises may have
    left the gantry anywhere, including partway across the deck, and the arc
    guard has to read that as unknown rather than as wherever it set off from.
    Pass ``None`` for a move that ends somewhere no labware names: a raw jog, a
    deck fixture, the trash, a home, or anything the gripper drove.
    """
    self._current_labware_id = None
    yield
    self._current_labware_id = labware_id

  def forget_pipetting_position(self) -> None:
    """Stop believing the gantry is still over its last labware.

    For when something OTHER than a move by this object put it somewhere else:
    an operator recovering the robot by hand, an e-stop. The arc guard has to
    read that as unknown, or the next pipetting move skips its traversal arc.
    """
    self._current_labware_id = None

  def attach_deck(self, deck: FlexDeck) -> None:
    """Swap in a new deck, so a caller can describe the deck without rebuilding
    the robot (and losing the link and the run with it).

    Clears the labware caches: their ids describe the deck being replaced, and
    serving one for the new deck would address the wrong slot.
    """
    self.deck = deck
    self._loaded_labware.clear()
    self._defined_labware.clear()
    self._stub_labware.clear()
    self._current_labware_id = None
    self._loaded_geometry.clear()

  async def _create_run(self) -> str:
    # labwareIds and uploaded definitions are both run-scoped server-side, so
    # a new run must not serve cached identities from a previous one.
    run_id = await super()._create_run()
    self._loaded_labware.clear()
    self._defined_labware.clear()
    self._stub_labware.clear()
    self._current_labware_id = None
    self._loaded_geometry.clear()
    return run_id

  async def _model_setup(self) -> None:
    """Discover and compose heads. Homing is setup()'s own step, so that a
    caller can ask what is mounted without moving the robot."""
    # Discover ALL mounted pipettes (not just the first — _discover_pipette
    # only surfaces one) and compose the matching head per mount. The base
    # setup() no longer discovers/loads a pipette itself (that would double
    # `loadPipette` the first mount), so this is the only place a Flex loads
    # its pipettes.
    # Discovery is re-runnable: drop whatever a previous setup composed rather
    # than stacking a second set of heads onto dead pipette ids.
    self.left = self.right = self.head96 = None
    self.gripper = None
    self._heads.clear()

    instruments_data = await self._get_instruments()
    pipettes = self._parse_pipettes(instruments_data)

    if not pipettes:
      raise OpentronsError("No pipette detected", f"{self.host}:{self.port}")

    if any(pip.channels == 96 for pip in pipettes) and len(pipettes) > 1:
      raise OpentronsError(
        "Impossible instrument combination",
        "A 96-channel head cannot be mounted alongside another pipette on a Flex.",
      )

    for pip in pipettes:
      pipette_id = await self._load_pipette(pip.pipette_name, pip.mount)
      head_cls = _CHANNELS_TO_HEAD.get(pip.channels)
      if head_cls is None:
        raise OpentronsError(
          "Unsupported pipette channel count",
          f"{pip.channels} channels (mount '{pip.mount}') has no matching FlexHead.",
        )
      head = head_cls(self, pip.mount, pipette_id, pip.channels, pip.pipette_model, pip.max_volume)

      if pip.channels == 96:
        self.head96 = head
      elif pip.mount == "left":
        self.left = head
      elif pip.mount == "right":
        self.right = head
      else:
        raise OpentronsError("Unknown mount", f"mount '{pip.mount}' is neither 'left' nor 'right'.")
      self._heads.append(head)

    for head in self._heads:
      await head._on_setup()

    # The gripper (extension mount) is optional: compose it when discovery
    # reports one, leave ``self.gripper`` None otherwise.
    gripper_model = self._parse_gripper(instruments_data)
    if gripper_model is not None:
      self.gripper = FlexGripper(self, gripper_model)
      logger.info("Discovered gripper on the extension mount (model: %s)", gripper_model)

  def _parse_gripper(self, instruments_data: Dict[str, Any]) -> Optional[str]:
    """Parse the /instruments response for a mounted gripper.

    Returns the gripper's model string, or ``None`` when none is mounted.
    Separate from ``_parse_pipettes``, which filters to
    ``instrumentType == 'pipette'`` and knows nothing about grippers.
    """
    for instrument in instruments_data.get("data", []):
      if instrument.get("instrumentType") == "gripper":
        return cast(str, instrument.get("instrumentModel", "unknown"))
    return None

  async def stop(self) -> None:
    # Drop any mounted tips to the trash BEFORE parking/disconnecting, so the
    # robot is never left holding tips. A failure here must not block the
    # home/cancel/disconnect that follows.
    try:
      trash: Optional[Trash] = self.deck.get_trash_area()
    except ValueError:
      trash = None
    if trash is not None:
      for head in reversed(self._heads):
        try:
          if any(tip is not None for tip in head.get_mounted_tips()):
            await head.discard_tips(trash)
        except Exception:
          logger.warning(
            "Dropping tips on stop failed for the %s head; continuing to disconnect.",
            head.mount,
            exc_info=True,
          )
    for head in reversed(self._heads):
      await head._on_stop()
    await super().stop()  # homes the gantry, then cancels the run + disconnects

  async def _ensure_labware_loaded(
    self,
    resource: Resource,
    *,
    allow_stub: bool = False,
    grip_distance_from_top: Optional[float] = None,
  ) -> str:
    """Load labware into the Flex run if not already loaded.

    ``allow_stub`` opts into the non-pipettable movable stub definition for a
    resource no real definition can be built from (a lid, an adapter, a tube
    rack). Only the gripper passes it: the stub's single fake well is
    somewhere to grip, not somewhere to pipette, and a zero-depth well at the
    labware's own bottom would put a tip on the deck. Every other caller gets
    an ``OpentronsError`` for such a resource, before any wire command --
    including when the gripper already loaded it earlier in the run.

    ``grip_distance_from_top`` feeds an uploaded custom definition's grip
    height and is honored on the FIRST load only; a cache hit (already
    loaded, or definition already uploaded) reuses the stored identity
    unchanged, and labware resolving to an official Opentrons load name
    ignores it entirely (the robot's own definition owns the grip height).
    Every discard is logged.
    """
    name = getattr(resource, "name", str(resource))
    if not allow_stub and name in self._stub_labware:
      raise _not_pipettable_error(resource)
    if name in self._loaded_labware:
      _warn_grip_distance_discarded(
        name,
        grip_distance_from_top,
        "it is already loaded in this run, and the grip height rides the definition it "
        "was loaded with",
      )
      return self._loaded_labware[name]

    slot = self.deck.get_slot(resource)
    if slot is None:
      raise OpentronsError(
        "Resource not on deck",
        f"'{name}' is not on a deck slot. Use deck.assign_child_at_slot(resource, slot='C1').",
      )

    try:
      load_name, version = self._ot_declared_identity(resource)
    except OpentronsError:
      # No official Opentrons definition: build one from the resource's PLR
      # geometry, upload it, and load by the uploaded definition's identity.
      namespace, load_name, version = await self._define_custom_labware(
        resource, grip_distance_from_top, allow_stub
      )
    else:
      namespace = _OT_NAMESPACE
      _warn_grip_distance_discarded(
        name,
        grip_distance_from_top,
        f"it loads the robot's own definition for '{load_name}', whose grip height is "
        "the vendor's to state (the robot grips at mid-height when it states none)",
      )
    # The robot assigns the id. Proposing one here would only make the
    # request body differ run to run, which is what breaks a replayed capture.
    result = await self._execute_command(
      "loadLabware",
      {
        "loadName": load_name,
        "location": slot_wire_location(slot),
        "namespace": namespace,
        "version": version,
        "displayName": name,
      },
    )
    labware_id = result.get("result", {}).get("labwareId")
    if not isinstance(labware_id, str):
      raise OpentronsError(
        "Labware load returned no id",
        f"loadLabware for '{name}' succeeded but reported no labwareId to address it by.",
      )

    self._loaded_labware[name] = labware_id
    definition = result.get("result", {}).get("definition")
    if isinstance(definition, dict):
      self._loaded_geometry[name] = _LoadedGeometry(
        wells=cast(Dict[str, Dict[str, Any]], definition.get("wells", {})),
        corner_offset_z=_corner_offset_z(definition),
        offset_id=cast(Optional[str], result.get("result", {}).get("offsetId")),
      )
    logger.info(
      "Loaded labware '%s' at slot %s -> ID: %s (OT: %s)",
      name,
      slot,
      labware_id,
      load_name,
    )
    return labware_id

  def _note_labware_offset(self, name: str, result: Dict[str, Any]) -> None:
    """Carry the offset a command reported onto the cached geometry.

    A reload or a gripper move re-places the labware and hands back the offset
    the robot applied. Left uncarried, the offset recorded at load time keeps
    saying "none applied" and a floor measured against the definition is wrong
    by exactly the offset nobody noticed.
    """
    loaded = self._loaded_geometry.get(name)
    if loaded is None:
      return
    offset_id = cast(Optional[str], result.get("result", {}).get("offsetId"))
    self._loaded_geometry[name] = loaded._replace(offset_id=offset_id)

  def well_bottom_deck_z(self, resource: Resource, well_name: str) -> float:
    """Deck z of the floor one well's ``liquid_height`` is measured from.

    The definition the RUN loaded owns this number. A labware loaded by an
    official Opentrons name is placed by vendor geometry PyLabRobot never sees,
    so deriving the floor from the PLR resource would answer in a frame the
    robot does not share -- and a probe reports a deck z, which is only useful
    once it is put back into the frame a caller pipettes in.
    """
    if resource.name in self._stub_labware:
      raise OpentronsError(
        "Labware has no pipettable geometry",
        f"'{resource.name}' was uploaded as a stub because it has no wells to pipette "
        "into, so the definition the run holds says nothing about where liquid sits.",
      )
    loaded = self._loaded_geometry.get(resource.name)
    if loaded is None:
      raise OpentronsError(
        "Labware geometry unknown",
        f"the run reported no definition for '{resource.name}', so the floor its liquid "
        "heights are measured from is unknown.",
      )
    if loaded.corner_offset_z is None:
      raise OpentronsError(
        "Labware geometry unknown",
        f"the definition loaded for '{resource.name}' says how far it sits above its slot "
        "in a schema this driver does not read, so a floor derived from it would be wrong "
        "by that distance.",
      )
    if loaded.offset_id is not None:
      raise OpentronsError(
        "Labware offset unaccounted for",
        f"the run applied labware offset {loaded.offset_id!r} to '{resource.name}', which "
        "moves it off where its definition puts it, so a height measured against that "
        "definition would be wrong by the offset.",
      )
    entry = loaded.wells.get(well_name)
    if entry is None:
      raise OpentronsError(
        "Unknown well",
        f"the definition loaded for '{resource.name}' has no well {well_name!r}.",
      )
    if "z" not in entry:
      raise OpentronsError(
        "Incomplete well definition",
        f"the definition loaded for '{resource.name}' gives well {well_name!r} no z.",
      )
    return resource.get_absolute_location(z="b").z + loaded.corner_offset_z + float(entry["z"])

  async def sync_tips_to_robot(self, tip_rack: TipRack) -> None:
    """Make the robot's tip-rack model agree with PyLabRobot's.

    The robot keeps its own record of which spots in a tip rack still hold a
    tip, and it is only ever changed by pickups and drops it performed itself.
    So anything that changes the rack outside a run -- a human taking tips, a
    rack swapped for a partly-used one, state restored from a database --
    leaves the two records disagreeing, with no symptom until the robot picks
    up from a spot it believes is full.

    This pushes PyLabRobot's record onto the robot in one ``setTipState`` per
    state, taking the resource tree as the truth. Set the tip layout first
    (``tip_rack.set_tip_state(...)``), then call this.

    There is no equivalent for liquid. ``loadLiquid`` is the robot's only way
    in, and it refuses any liquid the run did not already define, which a run
    created without a protocol never does. Well volumes therefore live only in
    the resource tree.
    """
    labware_id = await self._ensure_labware_loaded(tip_rack)
    # The robot grades a tip-rack well "clean"/"used"/"empty". PyLabRobot only
    # records whether a tip is there, so an occupied spot maps to "clean".
    occupied: List[str] = []
    empty: List[str] = []
    for spot in tip_rack.get_all_items():
      (occupied if spot.has_tip() else empty).append(tip_rack.get_child_identifier(spot))

    for well_names, state in ((occupied, "clean"), (empty, "empty")):
      if not well_names:
        continue
      await self._execute_command(
        "setTipState",
        {"labwareId": labware_id, "wellNames": well_names, "tipWellState": state},
      )
    logger.info(
      "Synced '%s' tip state to the robot: %d with a tip, %d empty",
      tip_rack.name,
      len(occupied),
      len(empty),
    )

  async def labware_moved_off_deck(self, resource: Resource) -> None:
    """Tell the robot an EXTERNAL agent (human or lab transporter) removed labware.

    A logical move, not a gripper motion: ``moveLabware`` with strategy
    ``manualMoveWithoutPause`` drops the labware from the robot-server's deck
    model, freeing its slot. Without this the slot stays occupied server-side
    and a later load into it fails with ``LocationIsOccupiedError``. The
    PLR-side deck slot is freed too. No wire command is sent for labware that
    was never loaded into the run; a re-add later loads fresh at its new slot
    and re-uploads its definition -- a different same-named resource must not
    inherit the departed labware's geometry.
    """
    name = getattr(resource, "name", str(resource))
    if name in self._loaded_labware:
      await self._execute_command(
        "moveLabware",
        {
          "labwareId": self._loaded_labware[name],
          "newLocation": "offDeck",
          "strategy": "manualMoveWithoutPause",
        },
      )
      del self._loaded_labware[name]
    self._defined_labware.pop(name, None)
    self._loaded_geometry.pop(name, None)
    self._stub_labware.discard(name)
    slot = self.deck.get_slot(resource)
    if slot is not None:
      self.deck.unassign_child_at_slot(slot)
    logger.info("Labware '%s' marked moved off-deck", name)

  async def reload_labware(self, resource: Resource) -> None:
    """Re-read the labware's position after a human moved it in its own slot.

    For labware nudged or re-seated where it already sits: the robot re-applies
    its labware offset, keeping the slot. Use ``gripper.move_labware`` to move
    it to another slot, and ``labware_moved_off_deck`` when it leaves the deck.
    """
    labware_id = await self._ensure_labware_loaded(resource)
    result = await self._execute_command("reloadLabware", {"labwareId": labware_id})
    self._note_labware_offset(resource.name, result)

  # --- Robot-level commands: axis motion, status surfaces, run log ---

  async def home(self) -> Dict[str, Any]:
    """Home all axes; the gantry parks at the rear-left-top, over no labware."""
    async with self._moving_to(None):
      return await super().home()

  async def move_axes_to(
    self,
    axis_map: Dict[str, float],
    critical_point: Optional[Dict[str, float]] = None,
    speed: Optional[float] = None,
  ) -> Dict[str, float]:
    """Move the named axes to absolute deck-frame positions (mm), returning where they land.

    Every axis the map does not name holds its current position.
    ``critical_point`` offsets the point on the mount that is driven to those
    positions, also as a per-axis map. ``speed`` is in mm/s (robot default if
    None).

    Raises:
      ValueError: If any axis is not one the robot addresses. Raised before
        any wire command is sent.
    """
    _require_robot_commands("robot/moveAxesTo", self.api_version)
    params = _axis_motion_params(axis_map, speed, critical_point)
    async with self._moving_to(None):
      return _reported_axis_position(await self._execute_command("robot/moveAxesTo", params))

  async def move_axes_relative(
    self, axis_map: Dict[str, float], speed: Optional[float] = None
  ) -> Dict[str, float]:
    """Jog the named axes by distances (mm) from where they are, returning where they land.

    Raises:
      ValueError: If any axis is not one the robot addresses. Raised before
        any wire command is sent.
    """
    _require_robot_commands("robot/moveAxesRelative", self.api_version)
    params = _axis_motion_params(axis_map, speed)
    async with self._moving_to(None):
      return _reported_axis_position(await self._execute_command("robot/moveAxesRelative", params))

  async def retract_axis(self, axis: str) -> None:
    """Retract ``axis`` to its home position, clearing the deck below it.

    Raises:
      ValueError: If ``axis`` is not one the robot addresses. Raised before
        any wire command is sent.
    """
    _validate_axes([axis])
    async with self._moving_to(None):
      await self._execute_command("retractAxis", {"axis": axis})

  async def set_status_bar(self, animation: str) -> None:
    """Play a built-in light-bar animation: "idle", "confirm", "updating", "disco" or "off".

    An animation is all the status bar takes: the robot owns the colours and
    the timing, so there is nothing per-light to address.
    """
    await self._execute_command("setStatusBar", {"animation": animation})

  async def set_rail_lights(self, on: bool) -> None:
    """Turn the deck rail lights on or off."""
    await self._execute_command("setRailLights", {"on": on})

  async def add_comment(self, message: str) -> None:
    """Record ``message`` in the run's command log; the robot does nothing else with it."""
    await self._execute_command("comment", {"message": message})

  async def wait_for_duration(self, seconds: float) -> None:
    """Hold the run for ``seconds`` before the next command runs."""
    # The command only completes once the robot finishes waiting, so the poll
    # gets the wait itself plus the usual command headroom.
    await self._execute_command(
      "waitForDuration", {"seconds": seconds}, timeout=seconds + COMMAND_POLL_HEADROOM
    )

  @staticmethod
  def _ot_declared_identity(resource: Resource) -> Tuple[str, int]:
    """The load name and version to load a resource by, if it declares one.

    Declaring ``ot_load_name`` is the whole rule: a resource that does one is
    loaded by name, and a resource that does not gets a definition built from
    its own geometry and uploaded. Whether the robot can actually resolve a
    declared name is the ROBOT's to answer, not ours. It looks in the
    definitions its own software shipped with AND in the custom labware a lab
    has added to it, so no list on this side can be right for every robot. A
    name it cannot resolve fails the ``loadLabware``, which surfaces where a
    caller expects it: at the point the labware goes onto the deck.

    ``ot_version`` picks the revision, and which revisions a robot holds is the
    robot's business too, so this defaults to 1 rather than guessing higher.
    Revision 1 is the one every definition has. It is also the OLDEST, and for
    much of Opentrons' own catalogue it predates the Flex and states no gripper
    grip height, so the robot grips at the labware's mid-height rather than
    where the vendor says. A later revision usually fixes that while leaving the
    well geometry alone, which is why the resources PyLabRobot ships for those
    plates declare one. Naming a revision the robot does not hold fails the
    load, so raising the default here would break older robots to grip better on
    newer ones.
    """
    declared = getattr(resource, "ot_load_name", None)
    if declared is None:
      raise OpentronsError(
        "No Opentrons load name declared",
        f"'{resource.name}' carries no ot_load_name, so there is no name to load it by.",
      )

    load_name = cast(str, declared)
    version = getattr(resource, "ot_version", None)
    if version is None:
      version = _OT_VERSION
    return load_name, cast(int, version)

  async def _define_custom_labware(
    self,
    resource: Resource,
    grip_distance_from_top: Optional[float] = None,
    allow_stub: bool = False,
  ) -> Tuple[str, str, int]:
    """Upload a geometry-derived definition for labware with no official Opentrons definition.

    Returns the uploaded definition's (namespace, load_name, version), parsed
    from the robot-server's ``definitionUri`` so the subsequent ``loadLabware``
    references exactly what the server stored. The parsed identity is cached
    (separately from ``_loaded_labware``) until the labware leaves the deck or
    a new run starts, so repeat calls within a stay on deck skip the
    re-upload -- which also means ``grip_distance_from_top`` only shapes the
    FIRST upload.
    """
    name = resource.name
    if name in self._defined_labware:
      _warn_grip_distance_discarded(
        name,
        grip_distance_from_top,
        "its custom definition was already uploaded in this run, with the grip height it "
        "carried then",
      )
      return self._defined_labware[name]

    definition = self._build_labware_definition(resource, grip_distance_from_top, allow_stub)
    assert self.run_id is not None, "No active run. Call setup() first."
    try:
      data = await self._post(f"/runs/{self.run_id}/labware_definitions", {"data": definition})
    except Exception as wire_error:
      await self._reraise_run_aware(wire_error)
    uri = cast(str, data["data"]["definitionUri"])
    namespace, load_name, version = uri.split("/")
    self._defined_labware[name] = (namespace, load_name, int(version))
    if not _has_pipettable_geometry(resource):
      self._stub_labware.add(name)
    logger.info("Uploaded custom labware definition for '%s': %s", name, uri)
    return self._defined_labware[name]

  @staticmethod
  def _build_labware_definition(
    resource: Resource,
    grip_distance_from_top: Optional[float] = None,
    allow_stub: bool = False,
  ) -> dict:
    """Build the definition matching the resource's type, or raise for unbuildable labware.

    Pipettable types get real-geometry definitions. Any other resource (lid,
    adapter, tube rack, ...) has no wells the robot can pipette, so it builds
    only as the non-pipettable movable stub, which ``allow_stub`` opts into.
    Geometry an Opentrons definition cannot describe (a rotated plate, a
    cavity floor the resource never declared) is refused here, before any
    wire command, rather than uploaded for the robot to act on.
    """
    try:
      if isinstance(resource, Plate):
        return build_plate_definition(resource, grip_distance_from_top)
      if isinstance(resource, TipRack):
        return build_tip_rack_definition(resource, grip_distance_from_top)
      if isinstance(resource, Container):
        return build_container_definition(resource, grip_distance_from_top)
    except ValueError as e:
      raise OpentronsError("Cannot build an Opentrons labware definition", str(e)) from e
    if not allow_stub:
      raise _not_pipettable_error(resource)
    return build_movable_labware_definition(resource, grip_distance_from_top)
