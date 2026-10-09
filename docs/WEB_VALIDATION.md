# Web workspace validation

Validated on Windows with Python 3.13 on October 9, 2026.

- 22 automated tests passed, including the original scientific engine checks and new HTTP API integration tests.
- CSV execution tested with 300 synthetic rows through the HTTP API, including generated results and persistent metadata.
- Tested invalid input rejection, cross-origin POST rejection, queue limits, interrupted-job recovery, worker failure persistence, and HTML/JSON/CSV downloads.
- Ran a 10-trial random-search synthetic benchmark through the browser. The UI moved from running to completed and displayed measured held-out results.
- Ran the Optuna CLI with five trials and generated a complete portable report in `reports/verified-optuna/`.
- Checked browser experiment setup and search-history navigation. Browser console reported no JavaScript errors during setup.
- Python compilation and JavaScript syntax checks passed.

The original report test relied on Windows' default text encoding. It now reads generated UTF-8 files explicitly.

`requirements-web-tested.txt` records the installed environment for this version. The original `requirements-tested.txt` and reference report are retained as prototype evidence, not the web app's installation lock.

This validation does not establish deployment safety, robustness outside the tested scenario, public hosting readiness, or support for external models. See the original protocol and limitations in README.md.
