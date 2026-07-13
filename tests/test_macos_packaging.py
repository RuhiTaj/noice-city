from __future__ import annotations

import hashlib
import plistlib
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging" / "macos"
EXPECTED_VERSION = "0.1.1"
EXPECTED_BUILD = "3"
EXPECTED_MACOS = "14.0"
EXPECTED_FASTAPI = "0.139.0"
EXPECTED_MULTIPART = "0.0.32"
EXPECTED_STARLETTE = "1.3.1"
MINIMUM_PYTEST = "9.0.3"
MLX_SHA256 = "ea5a594355c89c0095eaba413fd39d4caa8642fa13432dfb0c9354d141046467"
MLX_METAL_SHA256 = "5b64b20ac24b0c401f489de01e8209edc4d372125201f19314e6f39e385322aa"


class MacOSPackagingPolicyTests(unittest.TestCase):
    def read(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_release_version_and_deployment_target_are_consistent(self) -> None:
        pyproject = self.read("pyproject.toml")
        build = self.read("packaging/macos/build-app.sh")
        verify = self.read("packaging/macos/verify-app.sh")
        with (PACKAGING / "templates" / "Info.plist").open("rb") as handle:
            plist = plistlib.load(handle)

        self.assertRegex(
            pyproject,
            re.compile(rf'^version = "{re.escape(EXPECTED_VERSION)}"$', re.M),
        )
        self.assertIn(f'VERSION="{EXPECTED_VERSION}"', build)
        self.assertIn(f'BUILD_NUMBER="{EXPECTED_BUILD}"', build)
        self.assertIn(f'MINIMUM_MACOS="{EXPECTED_MACOS}"', build)
        self.assertIn(f'EXPECTED_VERSION="{EXPECTED_VERSION}"', verify)
        self.assertIn(f'EXPECTED_BUILD="{EXPECTED_BUILD}"', verify)
        self.assertIn(f'MAXIMUM_MACOS="{EXPECTED_MACOS}"', verify)
        self.assertEqual(plist["CFBundleShortVersionString"], EXPECTED_VERSION)
        self.assertEqual(plist["CFBundleVersion"], EXPECTED_BUILD)
        self.assertEqual(plist["LSMinimumSystemVersion"], EXPECTED_MACOS)

        for package, version in (
            ("fastapi", EXPECTED_FASTAPI),
            ("python-multipart", EXPECTED_MULTIPART),
        ):
            self.assertIn(f'"{package}=={version}"', pyproject)
        for package, version in (
            ("fastapi", EXPECTED_FASTAPI),
            ("python-multipart", EXPECTED_MULTIPART),
            ("starlette", EXPECTED_STARLETTE),
        ):
            self.assertRegex(
                self.read("uv.lock"),
                re.compile(
                    rf'^name = "{re.escape(package)}"\nversion = "{re.escape(version)}"$',
                    re.M,
                ),
            )
        self.assertIn(f'"pytest>={MINIMUM_PYTEST},<10"', pyproject)

    def test_exact_macos_14_mlx_wheels_and_bundle_scan_are_mandatory(self) -> None:
        prepare = self.read("packaging/macos/prepare-runtime.sh")
        build = self.read("packaging/macos/build-app.sh")
        verify = self.read("packaging/macos/verify-app.sh")
        audit = self.read("packaging/macos/audit-macho.sh")
        lock = self.read("uv.lock")

        for digest in (MLX_SHA256, MLX_METAL_SHA256):
            self.assertIn(digest, prepare)
            self.assertIn(digest, verify)
            self.assertIn(digest, lock)
        self.assertIn("macosx_14_0_arm64", prepare)
        self.assertIn("macosx_14_0_arm64", verify)
        self.assertIn('"$SCRIPT_DIR/audit-macho.sh"', prepare)
        self.assertIn('"$SCRIPT_DIR/audit-macho.sh"', build)
        self.assertIn('"$SCRIPT_DIR/audit-macho.sh"', verify)
        self.assertIn('/usr/bin/find "$ROOT" -type f -print0', audit)
        self.assertIn('/usr/bin/vtool -arch "$architecture" -show-build', audit)
        self.assertIn('loads unsafe dylib path', audit)
        self.assertIn('has unsafe LC_RPATH', audit)
        self.assertIn('command == "LC_LOAD_DYLIB"', audit)
        self.assertIn('command == "LC_RPATH"', audit)
        self.assertIn('[[ "$platform" == "MACOS" ]]', audit)
        self.assertIn("path\\tarchitectures\\tslice\\tplatform\\tminimum_macos\\tsdk", audit)
        self.assertIn("EXPECTED_MACHO_REPORT_HEADER", verify)
        macho_ci = self.read("tests/test_macho_audit_macos.sh")
        self.assertIn("--sdk iphoneos", macho_ci)
        self.assertIn("targets 'IOS', expected MACOS", macho_ci)
        self.assertIn("unsafe absolute LC_RPATH unexpectedly passed", macho_ci)
        self.assertIn("unsafe absolute dylib load path unexpectedly passed", macho_ci)
        self.assertIn("unsafe bare-relative dylib load path unexpectedly passed", macho_ci)
        self.assertNotIn("26.2", "\n".join((prepare, build, verify, audit)))

    def test_release_notices_and_license_are_nonempty_and_pinned(self) -> None:
        for name in (
            "LICENSE",
            "NOTICE",
            "TRADEMARKS.md",
            "THIRD_PARTY_NOTICES.md",
            "MODEL_NOTICES.md",
        ):
            self.assertGreater((ROOT / name).stat().st_size, 100, name)
        self.assertEqual(
            hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest(),
            "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        )

    def test_packaging_shell_scripts_parse(self) -> None:
        for script in sorted(PACKAGING.glob("*.sh")):
            subprocess.run(["/bin/bash", "-n", str(script)], check=True)

    def test_release_dmg_verifies_source_and_mounted_app(self) -> None:
        verify = self.read("packaging/macos/verify-app.sh")
        build_dmg = self.read("packaging/macos/build-dmg.sh")
        self.assertIn('APP="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"', verify)
        self.assertIn('VERIFY_ARGS=(--release "$APP")', build_dmg)
        self.assertIn('MOUNT_VERIFY_ARGS=(--release "$MOUNTED_APP")', build_dmg)
        self.assertIn("hdiutil attach -readonly -nobrowse", build_dmg)

    def test_release_builder_requires_clean_committed_source_provenance(self) -> None:
        build = self.read("packaging/macos/build-app.sh")
        verify = self.read("packaging/macos/verify-app.sh")
        self.assertIn("status --porcelain --untracked-files=normal", build)
        self.assertIn("release builds require a clean Git worktree", build)
        self.assertIn('SOURCE_COMMIT="$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)"', build)
        self.assertIn("RELEASE_PROVENANCE/SOURCE_COMMIT.txt", build)
        self.assertIn("worktree_clean=true", build)
        self.assertIn('cmp -s "$PROJECT_ROOT/uv.lock" "$DEPENDENCY_RELEASE/uv.lock"', build)
        self.assertIn("SOURCE_PROVENANCE", verify)

    def test_verifier_checks_signatures_before_isolated_runtime_execution(self) -> None:
        verify = self.read("packaging/macos/verify-app.sh")
        signature_gate = verify.index("APP_IS_SIGNED=0")
        first_runtime = verify.index('RUNTIME_MACHINE="$(')
        self.assertLess(signature_gate, first_runtime)
        self.assertIn("local verification requires at least an ad-hoc signature", verify)
        self.assertIn("unset PYTHONPATH PYTHONHOME", verify)
        self.assertIn('"$RUNTIME" -I -B', verify)

        prepare = self.read("packaging/macos/prepare-runtime.sh")
        self.assertIn("unset PYTHONPATH PYTHONHOME", prepare)
        self.assertNotRegex(prepare, re.compile(r'"\$PYTHON" -(?!I\b)'))

        build = self.read("packaging/macos/build-app.sh")
        self.assertIn("unset PYTHONPATH PYTHONHOME", build)
        self.assertIn('runtime/bin/python3" -I -B', build)

    def test_verifier_rejects_escaping_packaged_symlinks(self) -> None:
        verify = self.read("packaging/macos/verify-app.sh")
        self.assertIn('/usr/bin/find "$APP/Contents" -type l -print0', verify)
        self.assertIn("packaged symlink has an absolute target", verify)
        self.assertIn("packaged symlink traverses a parent directory", verify)

    @unittest.skipUnless(sys.platform == "darwin", "requires macOS packaging tools")
    def test_verifier_behaviorally_rejects_escaping_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            contents = Path(temporary) / "Bad.app" / "Contents"
            contents.mkdir(parents=True)
            link = contents / "unsafe"
            for target, expected in (
                ("/tmp/outside", "packaged symlink has an absolute target"),
                ("../outside", "packaged symlink traverses a parent directory"),
            ):
                link.symlink_to(target)
                result = subprocess.run(
                    [str(PACKAGING / "verify-app.sh"), str(contents.parent)],
                    text=True,
                    capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
                link.unlink()

    def test_release_smoke_harness_parses(self) -> None:
        harness = PACKAGING / "smoke-workflow.py"
        source = harness.read_text(encoding="utf-8")
        compile(source, str(harness), "exec")
        self.assertIn('evaluation_status in {"failed", "rejected"}', source)
        self.assertIn('evaluation.get("reference_free")', source)
        self.assertIn('evaluation.get("all_intelligible")', source)
        self.assertIn("sample_rate != 24_000", source)
        self.assertIn('result["seed"] != "3407"', source)
        self.assertIn("--expected-sha256", source)


if __name__ == "__main__":
    unittest.main()
