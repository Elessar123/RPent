# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompts for each moment the recording planner stopped to look.

A planner does not localize once. It takes a coarse reading of the whole scene,
flies the arm over each object in turn, and asks a narrower question from close
range -- and the question changes with the situation. On the mug it was
recovering from a mask that had leaked onto the table; on the plate it was
finding the middle of a flat surface; on the pudding it was reading the label to
confirm it had the right box. Those are different questions, so they get
different wordings.

The object noun is the card's own, so these are templates over it rather than
new task knowledge.
"""

PROMPTS = {
    # ---- 1. Opening survey, agentview, everything in one frame ------------
    # Codex: segment each entity, check the overlay, then back-project four
    # pixels down the mask's midline. The recipe's phrase is the query.
    "survey": "{object}",
    # ---- 2. Close refinement, wrist, arm parked above the object ----------
    # Codex reads the wrist view and names the object's centre in pixels. From
    # here the object fills the frame, so the query can say where the arm is
    # rather than how the object looks.
    "refine": "the center of the {object} directly below the gripper",
    # ---- 3. Recovery when the reading fell off the object -----------------
    # Codex: "the point mask leaked to table, but an interior wall sample
    # gives ...". It re-picks a point on the object's own surface, away from
    # the rim the mask escaped through.
    "recover": "a point on the body surface of the {object}, not the table behind it",
    # ---- 4. Identity confirmation ----------------------------------------
    # Codex: "the wrist view reads 'CHOCOLATE PUDDING', confirming identity".
    # It is checking that the thing under the gripper is the thing it meant,
    # by something written or printed on it.
    "confirm": "the printed label on the {object}",
    # ---- 5. The top face, which is what a placement height comes from -----
    # Codex sampled "multiple top pixels" for the pudding's final anchor.
    "top": "the top surface of the {object}",
    # ---- 6. What is in the gripper ---------------------------------------
    # Codex: "sample its visible patterned body ... rejecting any
    # cavity/background points".
    "held": "the body of the {object} held in the gripper",
    # ---- 7. Where it ended up --------------------------------------------
    # Codex re-localizes a released object before deciding to nudge it.
    "settled": "the {object} resting on the table",
}


def build(situation: str, obj: str) -> str:
    return PROMPTS[situation].format(object=obj)


#: Which situation applies at each point in a recipe. This is the mapping from
#: Codex's procedure onto the actions a recipe actually contains.
SCHEDULE = """
recipe action                     situation   camera      what it answers
--------------------------------- ----------- ----------- ---------------------
(episode start)                   survey      agentview   coarse position
move_to above object, before pick  refine      wrist       precise position
  if refine lands off the object   recover     wrist       a point on the body
  if two objects could be confused confirm     wrist       which one this is
place destination, before descent  top         wrist       surface height
set_gripper after pi0_pick         held        wrist       offset from gripper
release                            settled     wrist       where it landed
"""
