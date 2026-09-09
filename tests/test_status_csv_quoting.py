"""Status-CSV writing must survive raw VASP/MPI error text.

Regression for the 2026-09-08 corruption: the status field carries unmodified
VASP/MPI output, which contains newlines AND literal double quotes -- MPI prints
``... on node a02 calling\n"abort". This may have caused ...`` on every rank abort.
The writers used to hand-format the row as ``f'{i},{rank},"{msg}"'``, so the field
ended at the first interior quote and the remaining thousands of lines of error text
were parsed as fresh records. That both invented rows (any line that was a bare
integer became "structure N errored") and hid real ones (a whole failed structure
vanished from every reader-derived count).
"""

import csv
import os

import pytest

from saddlemill.config import append_status_row, read_status_csv_rows


MPI_ABORT_TEXT = (
    "error: vasp in /scratch/jobs/shard/VASP_7_-1 returned an error: 1 stderr  "
    "-----------------------------------------------------------------------------\n"
    "|     internal error in: radial.F  at line: 752                               |\n"
    "|     internal error in RAD_INT: RHOPS /= RHOAE                               |\n"
    "-----------------------------------------------------------------------------\n"
    "prterun has exited due to process rank 52 with PID 0 on node a02 calling\n"
    '"abort". This may have caused other processes in the application to be\n'
    "terminated by signals sent by prterun.\n"
    "1\n1\n1\n1\n1\n1\n1\n1\n"
    "   Process name: [prterun-a02-3825839@1,4] Exit code:    1\n"
)


def _status_dir(tmp_path, method="DoubleMinimization"):
    d = tmp_path / f"{method}_status_csvs"
    d.mkdir()
    return d


class TestAppendStatusRow:
    def test_embedded_quotes_and_newlines_round_trip(self, tmp_path):
        """One write of MPI abort text reads back as exactly one row."""
        d = _status_dir(tmp_path)
        append_status_row(d / "status_rank_7.csv", [7, 7, -1, 7, 0, MPI_ABORT_TEXT])

        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert len(rows) == 1
        assert rows[0][:5] == ["7", "7", "-1", "7", "0"]
        assert rows[0][5] == MPI_ABORT_TEXT

    def test_no_fragment_rows_from_bare_integers(self, tmp_path):
        """The wall of bare '1' lines must not become 'structure 1 errored'."""
        d = _status_dir(tmp_path)
        append_status_row(d / "status_rank_7.csv", [7, 7, -1, 7, 0, MPI_ABORT_TEXT])

        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert [r[0] for r in rows] == ["7"]
        assert not any(len(r) == 1 for r in rows)

    def test_second_record_is_not_swallowed(self, tmp_path):
        """A failure written after a quote-bearing one stays visible.

        This is the half that hid lemat_e0 idx 10 from every count.
        """
        d = _status_dir(tmp_path)
        f = d / "status_rank_7.csv"
        append_status_row(f, [7, 7, -1, 7, 0, MPI_ABORT_TEXT])
        append_status_row(f, [7, 7, 1, 7, 0, MPI_ABORT_TEXT])
        append_status_row(f, [10, 7, -1, 10, 0, MPI_ABORT_TEXT])
        append_status_row(f, [10, 7, 1, 10, 0, MPI_ABORT_TEXT])

        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert len(rows) == 4
        assert sorted({r[0] for r in rows}) == ["10", "7"]

    def test_historical_on_disk_shape_preserved(self, tmp_path):
        """Numeric ids unquoted, status quoted -- same shape as before the fix."""
        d = _status_dir(tmp_path)
        f = d / "status_rank_0.csv"
        append_status_row(f, [3, 0, -1, 3, 45, "converged"])
        # read bytes: text mode would normalise line endings and hide a regression
        assert f.read_bytes() == b'3,0,-1,3,45,"converged"\n'

    def test_plain_statuses_round_trip(self, tmp_path):
        """Ordinary statuses are unaffected by the change."""
        d = _status_dir(tmp_path)
        f = d / "status_rank_0.csv"
        for side, steps, msg in [(-1, 45, "converged"), (1, 600, "not_converged")]:
            append_status_row(f, [3, 0, side, 3, steps, msg])

        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert [r[5] for r in rows] == ["converged", "not_converged"]

    @pytest.mark.parametrize("n_fields", [3, 4, 6])
    def test_variable_arity_methods(self, tmp_path, n_fields):
        """Minimization/SinglePoint write 3 fields, NEB 4, Dimer/Sella/DoubleMin 6."""
        d = _status_dir(tmp_path, "Dimer")
        fields = list(range(n_fields - 1)) + [MPI_ABORT_TEXT]
        append_status_row(d / "status_rank_0.csv", fields)

        rows = read_status_csv_rows("Dimer", str(tmp_path))
        assert len(rows) == 1
        assert len(rows[0]) == n_fields


class TestLegacyFilesStillRead:
    """The reader must keep working on files already on disk."""

    def test_clean_legacy_file_unchanged(self, tmp_path):
        d = _status_dir(tmp_path)
        (d / "status_rank_0.csv").write_text(
            '0,0,-1,0,45,"converged"\n'
            '0,0,1,0,56,"converged"\n'
            '1,0,-1,1,63,"not_converged"\n'
        )
        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert len(rows) == 3
        assert [r[5] for r in rows] == ["converged", "converged", "not_converged"]

    def test_legacy_multiline_status_without_quotes_in_text(self, tmp_path):
        """Hand-quoted multi-line text with no interior quote parsed fine before
        the fix, and must still parse fine after it."""
        d = _status_dir(tmp_path)
        (d / "status_rank_0.csv").write_text(
            '2,0,-1,2,0,"error: vasp returned an error: 1 stderr\n'
            'DAV:  1  -0.5E+03\n'
            'DAV:  2  -0.6E+03\n'
            '"\n'
        )
        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert len(rows) == 1
        assert rows[0][0] == "2"
        assert "DAV:  2" in rows[0][5]

    def test_corrupt_legacy_file_still_yields_its_readable_rows(self, tmp_path):
        """A file already shredded by the old writer is not made worse.

        The reader's fragment guard (drop rows whose first field is not an int)
        is unchanged; this pins that behaviour so the writer fix cannot silently
        alter how existing corrupt files are read.
        """
        d = _status_dir(tmp_path)
        (d / "status_rank_7.csv").write_text(
            '7,7,-1,7,0,"error: vasp ... calling\n'
            '"abort". This may have caused other processes\n'
            '1\n1\n1\n'
            'not-an-integer,junk\n'
            '"\n'
        )
        rows = read_status_csv_rows("DoubleMinimization", str(tmp_path))
        assert rows, "reader must still return the rows it can parse"
        assert all(r[0].strip().lstrip("-").isdigit() for r in rows)
