# Troubleshooting

??? failure "`RuntimeError: DefaultCPUAllocator: not enough memory`"
    The machine ran out of RAM. This happened on the original 8 GB laptop when two or three
    experiments ran at once.

    - Run experiments **one at a time** (chain them with `;` in PowerShell).
    - Close browsers and other heavy apps.
    - Re-run the same command. Sweeps and U3 **resume** from finished seeds/conditions.
    - Or move to a [free GPU](gpu.md).

??? failure "`ModuleNotFoundError: No module named 'torch'`"
    You're using the system Python, not the virtual environment. Always call
    `.venv\Scripts\python` (Windows) or `.venv/bin/python`.

??? failure "`UserWarning: Failed to initialize NumPy`"
    Harmless, but install numpy to silence it: `.venv\Scripts\python -m pip install numpy`.

??? failure "Garbled `±` or Assamese digits in the console"
    Windows console encoding. Set UTF-8 before running:
    ```powershell
    $env:PYTHONIOENCODING = "utf-8"
    ```

??? question "Accuracy is near 0 for a long time, then jumps"
    Normal for algorithmic tasks. Models often learn the output *format* first, then the
    algorithm, sometimes abruptly ("grokking"). Train longer before concluding anything.

??? question "Results vary a lot between seeds"
    At this scale they will. That's why every experiment reports a 95% CI over seeds.
    If CIs overlap, add seeds before claiming a difference.

??? question "Training is slow on CPU"
    Expected: ~9 minutes per 1,500 steps for the default model. Reduce `train.steps`,
    `model.d_model` or `eval.task_samples` for quick iteration, or use a GPU.
