"""Native-oracle building blocks for the immutable Dodge cartridge."""

from dodge.native.assets import (
    AssetExtractionError,
    extract_asset_bundle,
    validate_asset_bundle,
)
from dodge.native.batch import NativeBatchEnvironment, NativeBatchResult, NativeDodgeEnv
from dodge.native.compat import PicoCompat, PicoFixed, PicoInput, PicoRng
from dodge.native.compatibility import (
    build_compatibility_report,
    run_compatibility_report,
)
from dodge.native.manifest import (
    CartridgeManifest,
    FileIdentity,
    manifest_for_path,
)
from dodge.native.oracle import OracleTrace, run_oracle_trace
from dodge.native.p2 import accept_p2_bundle, build_p2_acceptance_report
from dodge.native.raster import IndexedRaster, RasterError

__all__ = [
    "AssetExtractionError",
    "accept_p2_bundle",
    "build_p2_acceptance_report",
    "CartridgeManifest",
    "FileIdentity",
    "IndexedRaster",
    "NativeBatchEnvironment",
    "NativeBatchResult",
    "NativeDodgeEnv",
    "OracleTrace",
    "PicoCompat",
    "PicoFixed",
    "PicoInput",
    "PicoRng",
    "RasterError",
    "build_compatibility_report",
    "extract_asset_bundle",
    "manifest_for_path",
    "run_compatibility_report",
    "run_oracle_trace",
    "validate_asset_bundle",
]
