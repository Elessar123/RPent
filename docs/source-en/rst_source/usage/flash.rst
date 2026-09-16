Flash Mode
==========

**Flash Mode** is an evaluation-only execution mode that replays a plan stored
in memory. Each plan is derived from a simulator-verified successful episode
and contains an action sequence and visual anchors. During evaluation, RPent
locates those anchors in the current scene and adjusts the recorded waypoints.
It does not call an LLM to plan actions; SAM3, Molmo, and the VLA still perform
perception and low-level execution.

Select Flash Mode with ``--planner flash``. RPent evaluates by default;
combining Flash Mode with ``--explore`` is rejected before services start.
The current implementation supports LIBERO-PRO Spatial, Object, Goal, and Long
(``10``), across the task and swap suites.

LIBERO-PRO performance and execution time
-----------------------------------------

Across the complete 800-case LIBERO-PRO matrix (Spatial, Object, Goal, and Long;
task/swap; 10 seeds per task), Flash Mode solved 581 episodes (72.63%). Codex
without reasoning solved 500 (62.50%), while Codex with high reasoning solved
628 (78.50%). The two tasks without a successful source trace and therefore no
Flash plan are conservatively counted as 0/10.

.. image:: https://github.com/RLinf/misc/raw/main/rpent/flash/flash_libero_pro_performance_time.png
   :alt: Per-task success rate and execution-time comparison between Flash Mode and Codex on all LIBERO-PRO suites
   :width: 100%
   :align: center

The timing excludes model and service startup. Codex time is the mean planner
execution time over available records for each task. Flash Mode timing uses the
tool-execution time from the successful episode underlying each final plan
(one timing sample per plan). Success rates use the complete 800-case matrix for
every method. Both Codex baselines have planner duration records for all 800
cases.

How replay works
----------------

Each Flash plan contains an action sequence and a set of anchors. An anchor describes a
task-relevant object or location, such as the object to pick or the destination
for a placement. Actions that depend on an anchor store their offset from that
anchor instead of relying only on an absolute coordinate.

At run time, RPent extracts the anchors required by the plan and locates each
one with the interface recorded for it:

* **SAM3** locates segmentation anchors and returns an object mask and its
  position.
* **Molmo** locates point anchors by pointing to the requested object or
  location in the camera image.

RPent then combines each live anchor position with the offset stored in the
plan and executes the resulting waypoint. This lets the same plan run when
objects appear at different positions.

Flash plans in memory
---------------------

Flash plans are distributed through the `RLinf/RPent-memory Flash Mode directory
<https://huggingface.co/datasets/RLinf/RPent-memory/tree/main/libero/flash>`_
on Hugging Face rather than tracked in Git. With ``--memory-profile hf``
(the evaluation default), RPent synchronizes ``libero/flash/**`` into
``memory/libero/flash`` before execution.
The corpus contains 78 plans for 80 task identities. ``goal_swap_t0`` and
``10_swap_t9`` have no successful source plan and cannot use Flash Mode.

.. code-block:: text

   memory/libero/flash/
     object_swap_t3_anchors.json   objects and locations to locate at run time
     object_swap_t3_plan.json      actions and their anchor-relative coordinates

The task selects the plan. The seed changes the environment layout, not the
plan used for the task.

Generate a Flash plan
---------------------

The generator creates one plan from one simulator-verified successful episode.
Its two required inputs are the episode audit JSON and the matching primitive
recipe JSONL:

.. code-block:: bash

   python -m robots.libero.flash.generate \
     --audit results/goal_swap_t3_s7.json \
     --recipe results/goal_swap_t3_s7_recipe.jsonl \
     --destination memory/libero/flash

The audit must contain a non-empty ``task_language`` (or
``perturbed_task_language``) and ``libero_terminated: true``. The audit and
recipe filenames, plus the audit suite/task/seed fields when present, must
identify the same episode.

If ``segment_*.json`` readings were saved for the episode, pass their directory
with ``--segments``. Otherwise the generator derives semantic Molmo anchors from
the instruction and the recipe's ordered pick/release or articulation
transactions. Nearby ``move_to`` and ``move_pose`` coordinates are stored as XY
offsets from those anchors, in the format consumed by Flash Mode replay.

The relation parser supports all 80 LIBERO-PRO tasks: Spatial, Object, Goal, and
Long (``10``), across both task and swap suites. Long instructions are preserved
as ordered transactions, including dependent actions such as turning on the
stove before placement or closing an appliance after insertion.

To download only the Flash plans manually, run:

.. code-block:: bash

   hf download RLinf/RPent-memory --repo-type dataset \
     --include "libero/flash/**" --local-dir memory

Evaluate with Flash Mode
------------------------

Start Molmo first, then pass its endpoint to RPent:

.. code-block:: bash

   rpent --robot libero --planner flash \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

Flash replay supports the task and swap suites for LIBERO-PRO Spatial,
Object, Goal, and Long (``10``), for 80 task identities in total; execution requires an available plan.

The VLA and SAM3 services use the normal LIBERO runtime configuration. You can
also connect to services that are already running with ``--vla-endpoint`` and
``--sam3-endpoint``.

Use local memory
----------------

To evaluate with an existing local corpus, select it explicitly:

.. code-block:: bash

   rpent --robot libero --planner flash \
     --memory-profile local --memory-dir /path/to/memory/libero \
     --suite libero_object_swap --task 3 --seed 0 \
     --molmo-endpoint http://127.0.0.1:20703

The selected directory must contain ``flash/object_swap_t3_plan.json`` and
``flash/object_swap_t3_anchors.json``. Local mode never downloads memory from
Hugging Face. Evaluation reads the prepared plan without generating or updating
it. A missing or incomplete plan produces an error; execution does not fall
back to an LLM planner.

Molmo setup
-----------

Molmo requires a newer ``transformers`` version than the LIBERO policy
environment, so run it in a separate Python environment:

.. code-block:: bash

   uv venv --python 3.11 /path/to/molmo-venv
   /path/to/molmo-venv/bin/pip install -e ".[molmo]"

Download ``allenai/Molmo2-8B`` from `Hugging Face
<https://huggingface.co/allenai/Molmo2-8B>`_ or `ModelScope
<https://modelscope.cn/models/allenai/Molmo2-8B>`_, then start the service:

.. code-block:: bash

   export MOLMO_CHECKPOINT_PATH=/path/to/Molmo2-8B
   PYTHONPATH=/path/to/RPent /path/to/molmo-venv/bin/python \
     rpent/robots/components/molmo_server.py \
     --transport http --host 127.0.0.1 --port 20703

Replay multiple layouts
-----------------------

Run the same Flash plan with different seeds to evaluate it on different
layouts. Reusing existing VLA, SAM3, and Molmo services avoids loading the
models again for every run:

.. code-block:: bash

   for seed in $(seq 0 9); do
     rpent --robot libero --planner flash \
       --suite libero_object_swap --task 3 --seed "$seed" \
       --output-dir logs/sweep/swap_t3_s$seed \
       --vla-endpoint http://127.0.0.1:20701 \
       --sam3-endpoint http://127.0.0.1:20702 \
       --molmo-endpoint http://127.0.0.1:20703
   done
