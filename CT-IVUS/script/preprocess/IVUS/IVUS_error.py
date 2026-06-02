import argparse
from pathlib import Path

import pydicom


BASE_DATA_DIR = Path("/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT")
BASE_SAVE_DIR_ERROR = Path("/workspace/data/preprocessed/IVUS_csv")

_TARGETS = ["HT", "KH", "HK", "KY", "KN", "MN", "SH", "AC", "NT", "NTH", "WJ"]

_HT_PATIENTS_TO_SCAN = {"HT-014", "HT-021", "HT-026", "HT-036"}


class ErrorWriter:
	"""エラー行を `<name>\t<error>` 形式で追記するための薄いラッパ。"""

	def __init__(self, out_path: Path, overwrite: bool = False) -> None:
		self.out_path = out_path
		self.overwrite = overwrite
		self._fp = None

	def __enter__(self) -> "ErrorWriter":
		self.out_path.parent.mkdir(parents=True, exist_ok=True)
		mode = "w" if self.overwrite else "a"
		self._fp = self.out_path.open(mode, encoding="utf-8")
		return self

	def __exit__(self, exc_type, exc, tb) -> None:
		if self._fp is not None:
			self._fp.close()
			self._fp = None

	def write(self, name: str, error: str) -> None:
		if self._fp is None:
			raise RuntimeError("ErrorWriter is not opened")
		self._fp.write(f"{name}\t{error}\n")
		self._fp.flush()


def _check_dicom_pixel_array(dicom_path: Path, ew: ErrorWriter) -> None:
	"""dcmread と pixel_array 取得だけを試し、失敗したものだけ記録する。"""
	try:
		ds = pydicom.dcmread(dicom_path, force=True)
	except Exception as e:
		ew.write(str(dicom_path), f"DCMREAD:{type(e).__name__}")
		return

	try:
		_ = ds.pixel_array
	except Exception as e:
		ew.write(str(dicom_path), f"PIXEL_ARRAY:{type(e).__name__}")


# ---------- ターゲット別スキャン関数 ----------

def scan_ht(ew: ErrorWriter) -> None:
	"""HT: 患者ディレクトリ / "IVUS or OCT files" 直下のファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "HT"
	print(f"[HT] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue

		patient_id = patient_dir.name
		if patient_id not in _HT_PATIENTS_TO_SCAN:
			continue

		ivus_root = patient_dir / "IVUS or OCT files"
		if not ivus_root.exists():
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		files = [
			p for p in sorted(ivus_root.iterdir())
			if p.is_file() and not p.name.startswith('.')
		]
		if not files:
			ew.write(str(ivus_root), "NO_DICOM_FILES")
			continue

		for dicom_file in files:
			_check_dicom_pixel_array(dicom_file, ew)


def scan_kh(ew: ErrorWriter) -> None:
	"""KH: KH-IVUS-* / DATA.KH-** / <日付> / <日付> 直下の .dcm をチェック。"""
	src_dir = BASE_DATA_DIR / "KH"
	print(f"[KH] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for session_dir in sorted(src_dir.iterdir()):
		if not session_dir.is_dir() or session_dir.name.startswith('.'):
			continue
		if not session_dir.name.startswith("KH-IVUS"):
			continue

		for data_dir in sorted(session_dir.iterdir()):
			if not data_dir.is_dir() or data_dir.name.startswith('.'):
				continue
			if not data_dir.name.startswith("DATA.KH-"):
				continue

			for level1 in sorted(data_dir.iterdir()):
				if not level1.is_dir() or level1.name.startswith('.'):
					continue

				subdirs = [
					p for p in sorted(level1.iterdir())
					if p.is_dir() and not p.name.startswith('.')
				]
				leaf_dirs = subdirs if subdirs else [level1]

				for leaf_dir in leaf_dirs:
					dicom_files = [
						p for p in sorted(leaf_dir.iterdir())
						if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == ".dcm"
					]
					if not dicom_files:
						ew.write(str(leaf_dir), "NO_DICOM_FILES")
						continue
					for dicom_file in dicom_files:
						_check_dicom_pixel_array(dicom_file, ew)


def _scan_hk_patient_dir(patient_dir: Path, ew: ErrorWriter) -> None:
	session_dirs = [
		p for p in sorted(patient_dir.iterdir())
		if p.is_dir() and not p.name.startswith('.') and "IVUS" in p.name.upper()
	]
	if not session_dirs:
		ew.write(str(patient_dir), "NO_IVUS_DIR")
		return

	for session_dir in session_dirs:
		unnamed_dirs = [
			p for p in sorted(session_dir.iterdir())
			if p.is_dir() and not p.name.startswith('.') and p.name.lower().startswith("unnamed")
		]
		if not unnamed_dirs:
			ew.write(str(session_dir), "NO_UNNAMED_DIR")
			continue

		for unnamed_dir in unnamed_dirs:
			series_dirs = [
				p for p in sorted(unnamed_dir.iterdir())
				if p.is_dir() and not p.name.startswith('.')
			]
			if not series_dirs:
				ew.write(str(unnamed_dir), "NO_SERIES_DIR")
				continue

			for series_dir in series_dirs:
				dicom_files = [
					p for p in sorted(series_dir.iterdir())
					if p.is_file() and not p.name.startswith('.')
				]
				if not dicom_files:
					ew.write(str(series_dir), "NO_DICOM_FILES")
					continue
				for dicom_file in dicom_files:
					_check_dicom_pixel_array(dicom_file, ew)


def scan_hk(ew: ErrorWriter) -> None:
	"""HK: HK-N / HK-N(IVUS) / Unnamed / <シリーズ> 直下のファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "HK"
	print(f"[HK] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("HK-"):
			continue

		if patient_dir.name == "HK-8-2":
			for sub_dir in sorted(patient_dir.iterdir()):
				if not sub_dir.is_dir() or sub_dir.name.startswith('.'):
					continue
				if not sub_dir.name.startswith("HK-"):
					continue
				_scan_hk_patient_dir(sub_dir, ew)
		else:
			_scan_hk_patient_dir(patient_dir, ew)


def scan_ky(ew: ErrorWriter) -> None:
	"""KY: KY-N/IVUS または KY-N/①/IVUS 以下を再帰的にチェック。"""
	src_dir = BASE_DATA_DIR / "KY"
	print(f"[KY] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("KY-"):
			continue

		ivus_roots: list[Path] = []

		ivus_dir_direct = patient_dir / "IVUS"
		if ivus_dir_direct.exists() and ivus_dir_direct.is_dir():
			ivus_roots.append(ivus_dir_direct)

		for sub_dir in sorted(patient_dir.iterdir()):
			if not sub_dir.is_dir() or sub_dir.name.startswith('.') or sub_dir.name == "IVUS":
				continue
			ivus_under_sub = sub_dir / "IVUS"
			if ivus_under_sub.exists() and ivus_under_sub.is_dir():
				ivus_roots.append(ivus_under_sub)

		if not ivus_roots:
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		for ivus_root in ivus_roots:
			try:
				all_files = [
					p for p in ivus_root.rglob("*")
					if p.is_file() and not p.name.startswith('.')
				]
			except Exception:
				ew.write(str(ivus_root), "CANNOT_TRAVERSE")
				continue

			if not all_files:
				ew.write(str(ivus_root), "NO_DICOM_FILES")
				continue

			for dicom_file in sorted(all_files):
				_check_dicom_pixel_array(dicom_file, ew)


def scan_kn(ew: ErrorWriter) -> None:
	"""KN: KN-N / "KN-{N:02d} IVUS *" / DICOM 直下のファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "KN"
	print(f"[KN] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("KN-"):
			continue

		patient_raw = patient_dir.name
		try:
			prefix, num_str = patient_raw.split("-", maxsplit=1)
			num = int(num_str)
			ivus_prefix = f"{prefix}-{num:02d}"
		except Exception:
			print(f"  [WARN] Unexpected patient dir name: {patient_raw}")
			continue

		ivus_dirs = [
			p for p in sorted(patient_dir.iterdir())
			if p.is_dir() and p.name.startswith(f"{ivus_prefix} IVUS")
		]

		if not ivus_dirs:
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		for ivus_root in ivus_dirs:
			dicom_root = ivus_root / "DICOM"
			if not dicom_root.exists() or not dicom_root.is_dir():
				ew.write(str(ivus_root), "NO_DICOM_DIR")
				continue

			dicom_files = [
				p for p in sorted(dicom_root.iterdir())
				if p.is_file() and not p.name.startswith('.')
			]
			if not dicom_files:
				ew.write(str(dicom_root), "NO_DICOM_FILES")
				continue

			for dicom_file in dicom_files:
				_check_dicom_pixel_array(dicom_file, ew)


def scan_mn(ew: ErrorWriter) -> None:
	"""MN: 患者ディレクトリ直下の .dcm をチェック。"""
	src_dir = BASE_DATA_DIR / "MN"
	print(f"[MN] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("MN-"):
			continue

		dicom_files = [
			p for p in sorted(patient_dir.iterdir())
			if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == ".dcm"
		]
		if not dicom_files:
			ew.write(str(patient_dir), "NO_DICOM_FILES")
			continue

		for dicom_file in dicom_files:
			_check_dicom_pixel_array(dicom_file, ew)


def scan_sh(ew: ErrorWriter) -> None:
	"""SH: IVUS/DICOM/<シリーズ> 直下のファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "SH"
	print(f"[SH] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("SH-"):
			continue

		ivus_root = patient_dir / "IVUS" / "DICOM"
		if not ivus_root.exists():
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		for series_dir in sorted(ivus_root.iterdir()):
			if not series_dir.is_dir() or series_dir.name.startswith('.'):
				continue

			dicom_files = [
				p for p in sorted(series_dir.iterdir())
				if p.is_file() and not p.name.startswith('.')
			]
			if not dicom_files:
				ew.write(str(series_dir), "NO_DICOM_FILES")
				continue

			for dicom_file in dicom_files:
				_check_dicom_pixel_array(dicom_file, ew)


def scan_ac(ew: ErrorWriter) -> None:
	"""AC: "IVUS" を含むディレクトリを探索し最下層ファイルをチェック (OCT/OFDI 除外)。"""
	src_dir = BASE_DATA_DIR / "AC"
	print(f"[AC] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("AC-"):
			continue

		ivus_roots: list[Path] = []
		queue: list[Path] = [patient_dir]
		while queue:
			current = queue.pop(0)
			for p in sorted(current.iterdir()):
				if not p.is_dir() or p.name.startswith('.'):
					continue
				name_upper = p.name.upper()
				if "OCT" in name_upper or "OFDI" in name_upper:
					continue
				if "IVUS" in name_upper:
					ivus_roots.append(p)
				queue.append(p)

		if not ivus_roots:
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		for ivus_root in ivus_roots:
			candidate_dirs: list[Path] = []
			stack: list[Path] = [ivus_root]
			while stack:
				d = stack.pop()
				try:
					entries = list(d.iterdir())
				except Exception:
					continue
				files = [p for p in entries if p.is_file() and not p.name.startswith('.')]
				subdirs = [p for p in entries if p.is_dir() and not p.name.startswith('.')]
				if files:
					candidate_dirs.append(d)
				stack.extend(subdirs)

			if not candidate_dirs:
				ew.write(str(ivus_root), "NO_DICOM_FILES")
				continue

			candidate_dirs = sorted(set(candidate_dirs), key=lambda p: len(p.parts), reverse=True)
			leaf_dirs: list[Path] = []
			for d in candidate_dirs:
				is_ancestor = any(s.is_relative_to(d) for s in leaf_dirs)
				if not is_ancestor:
					leaf_dirs.append(d)

			for leaf_dir in sorted(leaf_dirs):
				try:
					files = [
						p for p in sorted(leaf_dir.iterdir())
						if p.is_file() and not p.name.startswith('.')
					]
				except Exception:
					ew.write(str(leaf_dir), "CANNOT_TRAVERSE")
					continue

				for dicom_file in files:
					name_upper = dicom_file.name.upper()
					if "OCT" in name_upper or "OFDI" in name_upper:
						continue
					_check_dicom_pixel_array(dicom_file, ew)


def scan_nt(ew: ErrorWriter) -> None:
	"""NT: IVUS/DICOM 以下を再帰的にチェック。"""
	src_dir = BASE_DATA_DIR / "NT"
	print(f"[NT] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("NT-"):
			continue

		ivus_root = patient_dir / "IVUS" / "DICOM"
		if not ivus_root.exists():
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		try:
			all_files = [
				p for p in ivus_root.rglob("*")
				if p.is_file() and not p.name.startswith('.')
			]
		except Exception:
			ew.write(str(ivus_root), "CANNOT_TRAVERSE")
			continue

		if not all_files:
			ew.write(str(ivus_root), "NO_DICOM_FILES")
			continue

		for dicom_file in sorted(all_files):
			_check_dicom_pixel_array(dicom_file, ew)


def scan_nth(ew: ErrorWriter) -> None:
	"""NTH: "NTH-*** IVUS" ディレクトリ直下の XA* ファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "NTH"
	print(f"[NTH] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("NTH-"):
			continue

		try:
			files = [
				p for p in sorted(patient_dir.iterdir())
				if p.is_file() and not p.name.startswith('.') and p.name.upper().startswith("XA")
			]
		except Exception:
			ew.write(str(patient_dir), "CANNOT_TRAVERSE")
			continue

		if not files:
			ew.write(str(patient_dir), "NO_DICOM_FILES")
			continue

		for dicom_file in files:
			_check_dicom_pixel_array(dicom_file, ew)


def scan_wj(ew: ErrorWriter) -> None:
	"""WJ: IVUS ディレクトリ直下の XA* ファイルをチェック。"""
	src_dir = BASE_DATA_DIR / "WJ"
	print(f"[WJ] source dir: {src_dir}")

	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("WJ-"):
			continue

		ivus_root = patient_dir / "IVUS"
		if not ivus_root.exists() or not ivus_root.is_dir():
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		try:
			files = [
				p for p in sorted(ivus_root.iterdir())
				if p.is_file() and not p.name.startswith('.') and p.name.upper().startswith("XA")
			]
		except Exception:
			ew.write(str(ivus_root), "CANNOT_TRAVERSE")
			continue

		if not files:
			ew.write(str(ivus_root), "NO_DICOM_FILES")
			continue

		for dicom_file in files:
			_check_dicom_pixel_array(dicom_file, ew)


# ---------- ディスパッチテーブル ----------

_SCAN_FUNCS = {
	"HT":  scan_ht,
	"KH":  scan_kh,
	"HK":  scan_hk,
	"KY":  scan_ky,
	"KN":  scan_kn,
	"MN":  scan_mn,
	"SH":  scan_sh,
	"AC":  scan_ac,
	"NT":  scan_nt,
	"NTH": scan_nth,
	"WJ":  scan_wj,
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"CCTI-calc-IVUS_OCT 内の IVUS を走査し、"
			"(1) IVUS ディレクトリが見つからない症例、(2) DICOM 読み込み失敗、(3) pixel_array 変換失敗 "
			"のみを txt に追記するスクリプト"
		),
	)
	parser.add_argument(
		"--target",
		"-t",
		choices=["ALL"] + _TARGETS,
		default="ALL",
		help="どのディレクトリ(HT/KH/HK/KY/KN/MN/SH/AC/NT/NTH/WJ)を処理するか",
	)
	parser.add_argument(
		"--output",
		"-o",
		default=None,
		help="出力 txt のパス。省略時は /workspace/data/preprocessed/IVUS_csv/IVUS_error_<target>.txt",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="出力 txt を上書きする (デフォルトは追記)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if args.output is None:
		out_path = BASE_SAVE_DIR_ERROR / f"IVUS_error_{args.target}.txt"
	else:
		out_path = Path(args.output)

	print(f"[ERROR-SCAN] target={args.target}")
	print(f"[ERROR-SCAN] output={out_path} (mode={'overwrite' if args.overwrite else 'append'})")

	with ErrorWriter(out_path, overwrite=args.overwrite) as ew:
		if args.target == "ALL":
			for t in _TARGETS:
				print(f"[ERROR-SCAN] scanning {t} ...")
				_SCAN_FUNCS[t](ew)
		else:
			_SCAN_FUNCS[args.target](ew)

	print("[ERROR-SCAN] done")


if __name__ == "__main__":
	main()
