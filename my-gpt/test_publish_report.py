import tempfile
from pathlib import Path
import unittest

from experiments.publish_report import (
    CapturedOutput,
    markdown_fence,
    publish_report,
    render_report,
    split_runner_metadata,
)


class PublishReportTests(unittest.TestCase):
    def test_preserves_multiline_unicode_stdout_and_extracts_metadata(self):
        stdout = "first line\nλ second line\n``embedded``\n\n"
        log = (
            stdout
            + "__RUN_TIME_SECONDS__ 12.34\n"
            + "__DEVICE__ cuda\n"
            + "__GPU_MAX_UTIL__ 97%\n"
            + "__GPU_MAX_MEM__ 852 MiB\n"
            + "__EXIT_CODE__ 0\n"
            + "__RUN_OK__\n"
        )

        captured = split_runner_metadata(log)

        self.assertEqual(captured.stdout, stdout)
        self.assertEqual(captured.metadata[-1], ("Run status", "ok"))
        report = render_report(30, "30_sparse_global_shortcut.py", captured)
        self.assertIn(stdout, report)
        self.assertIn("- Device: `cuda`", report)

    def test_embedded_backticks_use_a_longer_fence(self):
        self.assertEqual(markdown_fence("a ``` b ````` c"), "``````")

    def test_empty_output_is_valid(self):
        captured = split_runner_metadata("__EXIT_CODE__ 0\n__RUN_OK__\n")
        self.assertEqual(captured.stdout, "")
        self.assertIn("## Stdout\n\n```text\n```\n", render_report(1, "train.py", captured))

    def test_runner_like_text_inside_stdout_is_preserved(self):
        stdout = "__DEVICE__ is printed here as ordinary output\nstill running\n"
        log = stdout + "__EXIT_CODE__ 0\n__RUN_OK__\n"
        captured = split_runner_metadata(log)
        self.assertEqual(captured.stdout, stdout)

    def test_malformed_or_unknown_trailing_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no value"):
            split_runner_metadata("output\n__EXIT_CODE__\n")
        with self.assertRaisesRegex(ValueError, "unknown"):
            split_runner_metadata("output\n__EXIT_CODE__ 0\n__RUN_NOTE__ bad\n")

    def test_publish_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "run.log"
            output = root / "report.md"
            log.write_text("observed\n", encoding="utf-8")
            output.write_text("keep me", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                publish_report(
                    stage=1,
                    script="train.py",
                    log_path=log,
                    output_path=output,
                )
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

            publish_report(
                stage=1,
                script="train.py",
                log_path=log,
                output_path=output,
                force=True,
            )
            self.assertIn("observed\n", output.read_text(encoding="utf-8"))

    def test_rejects_invalid_stage_and_empty_script(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            render_report(0, "train.py", CapturedOutput("", ()))
        with self.assertRaisesRegex(ValueError, "script"):
            render_report(1, "", CapturedOutput("", ()))


if __name__ == "__main__":
    unittest.main()
