Task Cards
==========

A plan recorded as absolute waypoints cannot follow an object that moved. The
same plan recorded together with *what was localized* and *how far each
waypoint sat from that reading* can: the offset is task logic and survives a
change of layout, the coordinate is not.

A **task card** is a plan in that second form, recorded once from a solved
episode. Replaying it substitutes only perception: the actions, their order,
the prompts given to the policy and the gripper commands are all the card's,
and the grounder supplies the coordinates they run at.

The corpus
----------

Cards live in ``resources/libero/task_card``, beside the curated memory under
``resources/libero/memory``, and travel with the rest of that payload -- synced
from the HuggingFace resources dataset, or read from a copy already on disk.
Like everything under ``resources/``, the directory is not tracked in git.

There is one card per task, so nothing is chosen at run time: the task names
the card, and the card plus live grounding produces the trajectory.

.. code-block:: text

   resources/libero/task_card/
     index.json                 every card: task, source episode, instruction
     object/swap_t3/
       anchors.json             phrases localized, their readings, the anchor each yields
       plan.json                every action, with the anchor and offset behind its coordinate
       trace.md                 the same to read, with the recording planner's reasoning

No seed appears in the corpus. A card serves its task whatever layout it is
replayed against; the episode it was recorded from is kept inside the card as
``source``, as provenance rather than as a knob.

Molmo configuration
-------------------

Replay grounds hand-picked pixels with **Molmo**, served by
``rpent/robots/components/molmo_server.py``. Where SAM3 answers "which pixels
are this phrase", Molmo answers "where would you put the gripper" -- an
open-vocabulary point, for phrases no mask proposal names.

Molmo needs a newer ``transformers`` than the policy does, so it usually runs
under its own interpreter, which finds RPent through ``PYTHONPATH``:

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

Replaying one episode
---------------------

``--planner task_card`` is a planner backend like ``api`` or ``codex``, except
that the card decides the actions and no model is called. Everything else about
the run is unchanged:

.. code-block:: bash

   MOLMO_ENDPOINT=http://127.0.0.1:20703 \
     rpent --robot libero --planner task_card \
     --suite libero_object_swap --task 3 --seed 0

The corpus holds one card per task, so the seed selects the layout to solve,
never the plan used to solve it.

Replaying a whole sweep
-----------------------

``robots.libero.task_card.run`` drives many episodes without the CLI. It
connects to a policy, a segmenter and a grounder that are already serving, and
starts an environment server per episode:

.. code-block:: bash

   python -m robots.libero.task_card.run \
     --family object --tasks swap_t3 --seeds 0 1 2 \
     --vla-endpoint http://127.0.0.1:20701 \
     --sam3-endpoint http://127.0.0.1:20702 \
     --molmo-endpoint http://127.0.0.1:20703

Evaluation is single-attempt with no environment reset: a failed episode is
scored as failed, not retried from a clean state.

How a reading becomes a waypoint
--------------------------------

Each anchor is re-read **through the interface it was first read through**. A
``segment`` anchor is re-segmented, because its offsets are relative to a mask
centroid, and a wide container seen at an angle has that centroid some way from
where a pointing model points. Only pixels the recording agent chose by eye are
answered by the grounder.

Localization is two-stage, as the recording planner's was: a coarse survey of the
opening frame, then the arm parks over each hand-picked anchor and asks again
from the wrist, where the object fills the view. The close reading is kept only
when it agrees with the coarse one within 5 cm, so a wrist view that found
something else cannot overwrite a correct answer.

A wrist reading of a *held* object lands short of its centre, further the
taller the object. The correction is linear in the object's measured height.

Configuration
-------------

Both entry points take ``--help``. The options that matter:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Option
     - Meaning
   * - ``--family``
     - Perturbation family: ``object``, ``goal``, ``spatial`` or ``10``.
   * - ``--tasks``
     - Task keys, e.g. ``swap_t3 swap_t5``. Default: every card in the family.
   * - ``--seeds``
     - Seeds to replay each card against. Default: ``0``--``9``.
   * - ``--cards``
     - Card corpus. Defaults to ``resources/libero/task_card``.
   * - ``--output-dir``
     - Where per-episode output is written.
   * - ``--vla-endpoint`` / ``--sam3-endpoint`` / ``--molmo-endpoint``
     - The already-serving policy, segmenter and grounder
       (the sweep only).

``--planner task_card`` takes its grounder from ``MOLMO_ENDPOINT``, since the
planner is built before the robot runtime parses its own arguments.

Which recording became the card
-------------------------------

A card is one solved episode, and not every solved episode makes a good card.
Replaying all eight successful ``object/swap_t0`` recordings across ten seeds
gives 9/10 down to 1/10, and the two extremes have identical action counts,
pick counts and chunk budgets: solving a task once shows that the plan worked
on that layout, not that it transfers, and nothing in a plan's shape says which
it is.

So the card in the corpus is the recording that was measured to transfer, not
the one that looked best. That is a property of how the corpus was built,
settled before it shipped -- there is no card selection at run time.
