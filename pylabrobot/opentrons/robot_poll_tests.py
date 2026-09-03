"""Tests for how long a command is polled and when the wait is allowed to give up."""

import unittest
from typing import Any, Dict, Optional

from pylabrobot.opentrons.flex import OpentronsFlex
from pylabrobot.opentrons.robot import COMMAND_STATUS_POLL_INTERVAL, _command_budget
from pylabrobot.opentrons.transport import ChatterboxTransport
from pylabrobot.resources.opentrons.flex_deck import FlexDeck


class _ScriptedStatusTransport(ChatterboxTransport):
  """Answers command status from a script instead of succeeding on the first read.

  Counts the reads it served, and reports "running" until ``succeed_after`` reads
  have gone by (forever if None).
  """

  def __init__(self, succeed_after: Optional[int] = None) -> None:
    super().__init__()
    self.status_reads = 0
    self._succeed_after = succeed_after

  async def get(self, path: str) -> Dict[str, Any]:
    if "/commands/" not in path:
      return await super().get(path)
    self.status_reads += 1
    finished = self._succeed_after is not None and self.status_reads > self._succeed_after
    return {
      "data": {
        "id": path.rsplit("/", 1)[-1],
        "commandType": "",
        "status": "succeeded" if finished else "running",
        "result": {},
      }
    }


def _flex(transport: Optional[ChatterboxTransport] = None) -> OpentronsFlex:
  robot = OpentronsFlex(
    deck=FlexDeck(),
    host="localhost",
    transport=transport or ChatterboxTransport(),
  )
  robot.run_id = "run-1"
  return robot


class CommandBudgetTests(unittest.TestCase):
  """The budget arithmetic on its own, so a regression fails in milliseconds
  instead of sitting out the wait it wrongly imposed."""

  def test_a_command_with_no_plunger_travel_keeps_the_budget_it_was_given(self):
    """The headroom used to floor every command at 30 s, so a caller could not ask
    for a short wait on a command that does no pipetting."""
    self.assertEqual(_command_budget(0.01, {}), 0.01)

  def test_a_command_that_names_its_plunger_travel_gets_headroom_on_top(self):
    # 1000 uL at 10 uL/s is 100 s of travel, which no fixed budget covers.
    self.assertEqual(_command_budget(0.01, {"volume": 1000.0, "flowRate": 10.0}), 130.0)

  def test_the_poll_interval_is_a_fifth_of_a_second(self):
    self.assertEqual(COMMAND_STATUS_POLL_INTERVAL, 0.2)


class CommandPollTests(unittest.IsolatedAsyncioTestCase):
  async def test_a_command_that_finished_during_the_last_sleep_is_not_a_timeout(self):
    # The status read after the deadline is the one that sees it, and aborting a
    # move that already succeeded is worse than one extra read.
    transport = _ScriptedStatusTransport(succeed_after=1)
    robot = _flex(transport)

    result = await robot._execute_command("home", {}, timeout=0.01)

    self.assertEqual(result["status"], "succeeded")

  async def test_a_command_that_never_finishes_still_gives_up(self):
    robot = _flex(_ScriptedStatusTransport())

    with self.assertRaises(RuntimeError) as caught:
      await robot._execute_command("home", {}, timeout=0.01)

    self.assertIn("timed out", str(caught.exception))
