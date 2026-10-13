# reference_library

Every `.csv` / `.xy` / `.txt` / `.dat` pattern (two columns, `2theta intensity`) under this
folder appears in the app's sidebar as a tick box, grouped by sub-folder. Drop a file in and
refresh the page.

- `local/` is **git-ignored**. It holds patterns of mixed provenance (RRUFF downloads, patterns
  simulated from published CIFs) whose redistribution terms have not been checked, so they are not
  pushed to GitHub. Check the RRUFF terms and each CIF's licence before moving anything out of
  `local/` into a committed folder.
- The synthetic demo patterns in `demo_data/references/` are shown in the app as the group
  "demo (synthetic)" and are safe to ship.
