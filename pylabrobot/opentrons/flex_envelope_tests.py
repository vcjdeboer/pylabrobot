"""Tests for the native Flex operating envelope + its wiring into the heads.

``envelope.py`` derives every reach cap from grounded OT-3 primitives; ``checks.py``
is the pre-dispatch verification surface. These tests pin the derived caps to the
opentrons **8.8.1** shared-data the robot actually runs, and assert the computed
traversal plane replaces the hardcoded magic number.
"""

import asyncio
import unittest
from typing import List, Optional, Tuple

from pylabrobot.opentrons.checks import traversal_z
from pylabrobot.opentrons.envelope import FLEX_ENVELOPE
from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.flex_gripper import FlexGripper
from pylabrobot.opentrons.flex_head import FlexHead1, FlexHead8
from pylabrobot.opentrons.robot import OpentronsCommandError
from pylabrobot.opentrons.transport import ChatterboxTransport
from pylabrobot.resources import Plate, TipRack, cor_96_wellplate_360uL_Fb
from pylabrobot.resources.opentrons.flex_deck import FlexDeck


def _flex_head8() -> Tuple[OpentronsFlex, ChatterboxTransport, FlexHead8]:
  transport = ChatterboxTransport(pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")])
  flex = OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)
  asyncio.run(flex.setup())
  head = flex.left
  assert isinstance(head, FlexHead8)
  return flex, transport, head


def _commands_of(transport: ChatterboxTransport, command_type: str) -> List[dict]:
  return [c for c in transport.commands if c["commandType"] == command_type]


def _flex_with_two_plates(
  transport: ChatterboxTransport,
) -> Tuple[OpentronsFlex, TipRack, Plate, Plate]:
  """A set-up Flex with a tip rack in C1 and a full plate in each of C2 and D2."""
  from pylabrobot.resources.opentrons.flex_plates import flex_plate
  from pylabrobot.resources.opentrons.flex_tip_racks import flex_96_tiprack_50ul

  flex = OpentronsFlex(deck=FlexDeck(), host="localhost", transport=transport)
  asyncio.run(flex.setup())
  rack = flex_96_tiprack_50ul(name="rack")
  first = flex_plate("corning_96_wellplate_360ul_flat", name="plate")
  second = flex_plate("corning_96_wellplate_360ul_flat", name="other")
  flex.deck.assign_child_at_slot(rack, "C1")
  flex.deck.assign_child_at_slot(first, "C2")
  flex.deck.assign_child_at_slot(second, "D2")
  for plate in (first, second):
    for well in plate.get_all_items():
      well.tracker.set_volume(100.0)
  return flex, rack, first, second


def _gripper(flex: OpentronsFlex) -> FlexGripper:
  """The robot's gripper, asserted present."""
  assert flex.gripper is not None
  return flex.gripper


class _ArmableFailureTransport(ChatterboxTransport):
  """A ChatterboxTransport that fails one named command type while ``armed``.

  Lets a test drive the robot into position with every command succeeding, then
  arm the failure for the single command under test.
  """

  def __init__(self, fail_command: str, **kwargs) -> None:
    super().__init__(**kwargs)
    self.fail_command = fail_command
    self.armed = False

  async def post(self, path: str, json: Optional[dict] = None) -> dict:
    response = await super().post(path, json)
    sent = (json or {}).get("data", {})
    if self.armed and sent.get("commandType") == self.fail_command:
      # The returned dict is the same object the transport serves to the
      # status poll, so failing it here fails the command the caller awaits.
      response["data"]["status"] = "failed"
      response["data"]["error"] = {"errorType": "SimulatedFailure", "detail": "simulated"}
    return response


class TestRearCapGroundedTo881(unittest.TestCase):
  """The 8-channel rear reach cap is ``deck_extent_y + paddingOffsets.rear``.

  The rear padding is version-specific: opentrons 8.3.0 = -177.42, but the robot
  runs **8.8.1** where it is -169.42, giving a rear cap of 324.38. The stale
  8.3.0 value (316.38) must not be what the envelope carries.
  """

  def test_rear_cap_matches_881_shared_data(self):
    self.assertAlmostEqual(FLEX_ENVELOPE.padding_rear, -169.42)
    self.assertAlmostEqual(FLEX_ENVELOPE.rear_cap_y, 324.38)


class TestUnconditionalTiprackFloor(unittest.TestCase):
  """The travel plane never drops below a tip rack (99 + 10 margin = 109), even on
  a deck the model shows as empty or holding only short labware -- a rack that is
  present but unmodeled must still be cleared."""

  def test_empty_deck_still_clears_a_tiprack(self):
    self.assertAlmostEqual(traversal_z(FlexDeck()), 109.0)

  def test_short_labware_does_not_lower_the_floor(self):
    deck = FlexDeck()
    plate = cor_96_wellplate_360uL_Fb(name="plate")  # ~14 mm tall, well below a rack
    plate.ot_load_name = "corning_96_wellplate_360ul_flat"  # type: ignore[attr-defined]
    deck.assign_child_at_slot(plate, "C2")
    self.assertAlmostEqual(traversal_z(deck), 109.0)


class TestComputedTraversalPlane(unittest.TestCase):
  """A lateral jog defaults its ``minimumZHeight`` to the COMPUTED tip-safe plane
  (tallest labware top + arc margin), not a hardcoded 120.0 magic number."""

  def test_move_to_uses_computed_traversal_not_120(self):
    flex, transport, head = _flex_head8()
    try:
      plate = cor_96_wellplate_360uL_Fb(name="plate")
      plate.ot_load_name = "corning_96_wellplate_360ul_flat"  # type: ignore[attr-defined]
      flex.deck.assign_child_at_slot(plate, "C2")

      expected = traversal_z(flex.deck)
      self.assertNotAlmostEqual(expected, 120.0, msg="test needs labware whose plane != 120")

      asyncio.run(head.move_to(x=100.0, y=100.0, z=50.0))

      move_cmds = _commands_of(transport, "moveToCoordinates")
      self.assertEqual(len(move_cmds), 1)
      self.assertAlmostEqual(move_cmds[0]["params"]["minimumZHeight"], expected)
    finally:
      asyncio.run(flex.stop())


class TestTrashDropArcsHighEnough(unittest.TestCase):
  """The move to the trash after a dispense must arc at the computed traversal
  plane, not the engine's default -- otherwise it can travel too low and clip
  labware the robot was never told about."""

  def test_discard_tips_move_carries_computed_minimum_z_height(self):
    from pylabrobot.resources import cor_96_wellplate_360uL_Fb
    from pylabrobot.resources.opentrons.flex_tip_racks import flex_96_tiprack_50ul

    flex, transport, head = _flex_head8()
    try:
      rack = flex_96_tiprack_50ul(name="rack")
      plate = cor_96_wellplate_360uL_Fb(name="plate")
      plate.ot_load_name = "corning_96_wellplate_360ul_flat"  # type: ignore[attr-defined]
      flex.deck.assign_child_at_slot(rack, "C1")
      flex.deck.assign_child_at_slot(plate, "C2")
      trash = flex.deck.get_trash_area()

      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.discard_tips(trash))

      move = next(
        c for c in transport.commands if c["commandType"] == "moveToAddressableAreaForDropTip"
      )
      self.assertAlmostEqual(move["params"]["minimumZHeight"], traversal_z(flex.deck))
    finally:
      asyncio.run(flex.stop())


class TestBetweenSlotArcGuard(unittest.TestCase):
  """A pipetting move that crosses to a different slot is prefixed with a safe
  high moveToWell (>= the tip-rack floor); a move within the same labware is not."""

  def setUp(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(True)
    set_volume_tracking(True)

  def tearDown(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(False)
    set_volume_tracking(False)

  def _setup(self):
    from pylabrobot.resources import cor_96_wellplate_360uL_Fb
    from pylabrobot.resources.opentrons.flex_tip_racks import flex_96_tiprack_50ul

    flex, transport, head = _flex_head8()
    rack = flex_96_tiprack_50ul(name="rack")
    plate = cor_96_wellplate_360uL_Fb(name="plate")
    plate.ot_load_name = "corning_96_wellplate_360ul_flat"  # type: ignore[attr-defined]
    flex.deck.assign_child_at_slot(rack, "C1")
    flex.deck.assign_child_at_slot(plate, "C2")
    for w in plate.get_all_items():
      w.tracker.set_volume(100.0)
    return flex, transport, head, rack, plate

  def _setup_with_a_second_plate(self):
    """The same deck plus an ``other`` plate in D2, so a move can leave the plate
    the head pipetted over for a well on different labware."""
    transport = ChatterboxTransport(pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")])
    flex, rack, plate, other = _flex_with_two_plates(transport)
    head = flex.left
    assert isinstance(head, FlexHead8)
    return flex, transport, head, rack, plate, other

  def _move_to_wells(self, transport):
    return [c for c in transport.commands if c["commandType"] == "moveToWell"]

  def test_crossing_to_a_new_slot_arcs_high_first(self):
    flex, transport, head, rack, plate = self._setup()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))  # over the rack (C1)
      before = len(self._move_to_wells(transport))
      asyncio.run(head.aspirate(plate.column(0), volume=50))  # -> plate (C2), a new slot

      moves = self._move_to_wells(transport)
      self.assertEqual(
        len(moves), before + 1, "one safe move should precede the cross-slot aspirate"
      )
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
      # the safe move comes immediately before the aspirate
      types = [c["commandType"] for c in transport.commands]
      self.assertEqual(types[types.index("aspirate") - 1], "moveToWell")
    finally:
      asyncio.run(flex.stop())

  def test_moving_within_the_same_labware_does_not_arc_high(self):
    flex, transport, head, rack, plate = self._setup()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))  # cross-slot -> one safe move
      n = len(self._move_to_wells(transport))
      asyncio.run(head.dispense(plate.column(1), volume=50))  # same plate -> no new safe move
      self.assertEqual(len(self._move_to_wells(transport)), n, "within-slot move must not arc high")
    finally:
      asyncio.run(flex.stop())

  def test_a_move_to_a_deck_fixture_makes_the_next_pipetting_move_arc_high(self):
    """A trash or waste chute is not a slot's labware, so coming back to the plate
    crosses slots again even though it is the same plate it left."""
    flex, transport, head, rack, plate = self._setup()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      asyncio.run(head.move_to_addressable_area("movableTrashA3"))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(plate.column(1), volume=50))

      moves = self._move_to_wells(transport)
      self.assertEqual(
        len(moves),
        n + 1,
        "returning from a deck fixture must arc high, not descend from wherever it left",
      )
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
    finally:
      asyncio.run(flex.stop())

  def test_a_move_to_another_labwares_well_makes_the_next_pipetting_move_arc_high(self):
    """``move_to_well`` puts the head over whatever labware it names, so coming
    back to the plate it left crosses slots again."""
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      asyncio.run(head.move_to_well(other.get_item("A1")))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(plate.column(1), volume=50))

      moves = self._move_to_wells(transport)
      self.assertEqual(len(moves), n + 1, "coming back from another slot must arc high")
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
    finally:
      asyncio.run(flex.stop())

  def test_a_move_to_a_well_records_it_rather_than_forgetting_where_the_head_is(self):
    """The head really is over the labware it was told to move to, so pipetting
    there next needs no arc. Clearing the position instead would be safe but
    would buy a redundant full-height arc on every move-then-pipette."""
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      asyncio.run(head.move_to_well(other.get_item("A1")))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(other.column(0), volume=50))

      self.assertEqual(
        len(self._move_to_wells(transport)),
        n,
        "the head is already over that labware; pipetting there crosses nothing",
      )
    finally:
      asyncio.run(flex.stop())

  def test_a_relative_jog_makes_the_next_pipetting_move_arc_high(self):
    """A relative jog is the same arbitrary-position move ``move_to`` is: the
    head can be anywhere afterwards, including over another slot."""
    flex, transport, head, rack, plate = self._setup()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.move_relative("x", 120.0))
      asyncio.run(head.dispense(plate.column(1), volume=50))

      moves = self._move_to_wells(transport)
      self.assertEqual(len(moves), n + 1, "a jog leaves the head at an unknown position")
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
    finally:
      asyncio.run(flex.stop())

  def test_a_touch_tip_that_crosses_to_another_slot_arcs_high_first(self):
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.touch_tip(other, column=0))

      moves = self._move_to_wells(transport)
      self.assertEqual(len(moves), n + 1, "a touch on another slot's labware crosses slots")
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
      types = [c["commandType"] for c in transport.commands]
      self.assertEqual(types[types.index("touchTip") - 1], "moveToWell")
    finally:
      asyncio.run(flex.stop())

  def test_a_touch_tip_leaves_the_head_over_the_labware_it_touched(self):
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      asyncio.run(head.touch_tip(other, column=0))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(plate.column(1), volume=50))

      self.assertEqual(
        len(self._move_to_wells(transport)), n + 1, "coming back from the touch crosses slots"
      )
    finally:
      asyncio.run(flex.stop())

  def test_a_liquid_probe_that_crosses_to_another_slot_arcs_high_first(self):
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.try_liquid_probe(other, column=0))

      moves = self._move_to_wells(transport)
      self.assertEqual(len(moves), n + 1, "a probe on another slot's labware crosses slots")
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
      types = [c["commandType"] for c in transport.commands]
      self.assertEqual(types[types.index("tryLiquidProbe") - 1], "moveToWell")
    finally:
      asyncio.run(flex.stop())

  def test_a_liquid_probe_leaves_the_head_over_the_labware_it_probed(self):
    flex, transport, head, rack, plate, other = self._setup_with_a_second_plate()
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      asyncio.run(head.try_liquid_probe(other, column=0))
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(plate.column(1), volume=50))

      self.assertEqual(
        len(self._move_to_wells(transport)), n + 1, "coming back from the probe crosses slots"
      )
    finally:
      asyncio.run(flex.stop())


class TestPositionIsUnknownAfterAFailedMove(unittest.TestCase):
  """A move that raises may have left the head anywhere, including partway across
  the deck, so the tracked pipetting position has to end up unknown -- which is
  what makes the next pipetting move arc high."""

  def setUp(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(True)
    set_volume_tracking(True)

  def tearDown(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(False)
    set_volume_tracking(False)

  def _setup(self, fail_command: str):
    transport = _ArmableFailureTransport(
      fail_command, pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")]
    )
    flex, rack, plate, other = _flex_with_two_plates(transport)
    head = flex.left
    assert isinstance(head, FlexHead8)
    return flex, transport, head, rack, plate, other

  def _move_to_wells(self, transport):
    return [c for c in transport.commands if c["commandType"] == "moveToWell"]

  def test_a_jog_or_fixture_move_that_raises_leaves_the_position_unknown(self):
    movers = [
      ("moveToCoordinates", lambda head, other: head.move_to(x=100.0, y=100.0, z=50.0)),
      ("moveRelative", lambda head, other: head.move_relative("x", 120.0)),
      (
        "moveToAddressableArea",
        lambda head, other: head.move_to_addressable_area("movableTrashA3"),
      ),
      ("moveToWell", lambda head, other: head.move_to_well(other.get_item("A1"))),
    ]
    for command, call in movers:
      with self.subTest(command=command):
        flex, transport, head, rack, plate, other = self._setup(command)
        try:
          asyncio.run(head.pick_up_tips(rack, column=0))
          asyncio.run(head.aspirate(plate.column(0), volume=50))

          transport.armed = True
          with self.assertRaises(OpentronsCommandError):
            asyncio.run(call(head, other))
          transport.armed = False
          # After the failed command, so a mover whose own command IS a
          # moveToWell does not have it counted as the arc under test.
          n = len(self._move_to_wells(transport))

          asyncio.run(head.dispense(plate.column(1), volume=50))

          self.assertEqual(
            len(self._move_to_wells(transport)),
            n + 1,
            "a failed move must not leave the head trusting where it was",
          )
        finally:
          asyncio.run(flex.stop())

  def test_an_arc_that_raises_does_not_leave_the_head_trusting_where_it_was(self):
    """The high arc is itself a move. If it fails partway the head is somewhere
    over the deck, so the labware it started from is no longer where it is."""
    flex, transport, head, rack, plate, other = self._setup("moveToWell")
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))

      transport.armed = True
      with self.assertRaises(OpentronsCommandError):
        asyncio.run(head.dispense(other.column(0), volume=50))
      transport.armed = False
      n = len(self._move_to_wells(transport))

      asyncio.run(head.dispense(plate.column(1), volume=50))

      self.assertEqual(
        len(self._move_to_wells(transport)),
        n + 1,
        "a failed arc leaves the head between slots, not back at the plate",
      )
    finally:
      asyncio.run(flex.stop())

  def test_a_trash_drop_that_raises_at_the_trash_still_forgets_the_plate(self):
    """The worst case: the move to the trash succeeded and only the drop failed,
    so the head IS over the trash while the tracked position names the plate."""
    flex, transport, head, rack, plate, other = self._setup("dropTipInPlace")
    try:
      asyncio.run(head.pick_up_tips(rack, column=0))
      asyncio.run(head.aspirate(plate.column(0), volume=50))
      n = len(self._move_to_wells(transport))

      transport.armed = True
      with self.assertRaises(OpentronsCommandError):
        asyncio.run(head.discard_tips(flex.deck.get_trash_area()))
      transport.armed = False

      asyncio.run(head.dispense(plate.column(1), volume=50))

      self.assertEqual(
        len(self._move_to_wells(transport)),
        n + 1,
        "the head is over the trash; coming back to the plate must arc high",
      )
    finally:
      asyncio.run(flex.stop())


class TestEveryGantryMoveDropsTheTrackedPosition(unittest.TestCase):
  """One x/y gantry carries both pipette mounts and the gripper, so every command
  that can drive it laterally has to leave the pipetting position unknown. The
  arc guard reads that field, and a stale one skips the arc."""

  def setUp(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(True)
    set_volume_tracking(True)

  def tearDown(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(False)
    set_volume_tracking(False)

  def _move_to_wells(self, transport):
    return [c for c in transport.commands if c["commandType"] == "moveToWell"]

  def test_every_gantry_mover_makes_the_next_pipetting_move_arc_high(self):
    movers = [
      ("home", lambda flex, other: flex.home()),
      ("move_axes_to", lambda flex, other: flex.move_axes_to({"x": 200.0})),
      ("move_axes_relative", lambda flex, other: flex.move_axes_relative({"x": 20.0})),
      ("retract_axis", lambda flex, other: flex.retract_axis("x")),
      ("gripper.move_to", lambda flex, other: _gripper(flex).move_to(200.0, 200.0, 100.0)),
      ("gripper.move_labware", lambda flex, other: _gripper(flex).move_labware(other, "B2")),
    ]
    for name, call in movers:
      with self.subTest(mover=name):
        transport = ChatterboxTransport(
          pipettes=[("p50_multi_flex", 8, 1.0, 50.0, "left")], gripper=True
        )
        flex, rack, plate, other = _flex_with_two_plates(transport)
        head = flex.left
        assert isinstance(head, FlexHead8)
        try:
          asyncio.run(head.pick_up_tips(rack, column=0))
          asyncio.run(head.aspirate(plate.column(0), volume=50))
          n = len(self._move_to_wells(transport))

          asyncio.run(call(flex, other))
          asyncio.run(head.dispense(plate.column(1), volume=50))

          moves = self._move_to_wells(transport)
          self.assertEqual(
            len(moves), n + 1, "the gantry moved, so the head is no longer over the plate"
          )
          self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
        finally:
          asyncio.run(flex.stop())


class TestTheGantryIsSharedBetweenMounts(unittest.TestCase):
  """A Flex has one X/Y gantry and a Z per mount (the opentrons axis set is one
  ``X``, one ``Y``, then ``Z_L``/``Z_R``). Moving either head moves both, so the
  pipetting position belongs to the robot, not to one head."""

  def setUp(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(True)
    set_volume_tracking(True)

  def tearDown(self):
    from pylabrobot.resources import set_tip_tracking, set_volume_tracking

    set_tip_tracking(False)
    set_volume_tracking(False)

  def _setup(self):
    transport = ChatterboxTransport(
      pipettes=[
        ("p50_multi_flex", 8, 1.0, 50.0, "left"),
        ("p1000_single_flex", 1, 1.0, 1000.0, "right"),
      ]
    )
    flex, rack, a, b = _flex_with_two_plates(transport)
    left, right = flex.left, flex.right
    assert isinstance(left, FlexHead8) and isinstance(right, FlexHead1)
    return flex, transport, left, right, rack, a, b

  def _move_to_wells(self, transport):
    return [c for c in transport.commands if c["commandType"] == "moveToWell"]

  def test_a_move_by_one_head_makes_the_other_head_arc_high(self):
    flex, transport, left, right, rack, a, b = self._setup()
    try:
      asyncio.run(left.pick_up_tips(rack, column=0))
      asyncio.run(right.pick_up_tips(rack.get_item("A12")))
      asyncio.run(left.aspirate(a.column(0), volume=50))
      asyncio.run(right.aspirate(b.get_item("A1"), volume=50))
      n = len(self._move_to_wells(transport))

      asyncio.run(left.dispense(a.column(1), volume=50))

      moves = self._move_to_wells(transport)
      self.assertEqual(
        len(moves),
        n + 1,
        "the right head moved the gantry off plate a; the left head must arc back",
      )
      self.assertAlmostEqual(moves[-1]["params"]["minimumZHeight"], traversal_z(flex.deck))
    finally:
      asyncio.run(flex.stop())


if __name__ == "__main__":
  unittest.main()
