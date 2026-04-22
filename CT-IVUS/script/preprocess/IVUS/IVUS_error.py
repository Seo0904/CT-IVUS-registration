import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


BASE_DATA_DIR = Path("/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT")
# /workspace/data/preprocessed/IVUS/ の下に HT, KH, ... ごとのサブフォルダを作る想定 (PNG 用)
BASE_SAVE_DIR_PNG = Path("/workspace/data/preprocessed/IVUS")
# /workspace/data/preprocessed/IVUS_tiff/ の下に HT, KH, ... ごとのサブフォルダを作る想定 (TIFF 用)
BASE_SAVE_DIR_TIFF = Path("/workspace/data/preprocessed/IVUS_tiff")
BASE_SAVE_DIR_META = Path("/workspace/data/preprocessed/IVUS_meta")

# エラー一覧 (txt) の保存先
BASE_SAVE_DIR_ERROR = Path("/workspace/data/preprocessed/IVUS_csv")

# 実行時に main() 内で PNG / TIFF のどちらを使うかを切り替える
CURRENT_SAVE_DIR = BASE_SAVE_DIR_PNG


def save_dicom_array(img: np.ndarray, save_dir: Path, basename: str) -> None:
	"""view_IVUS.py と同様のルールで ndarray を PNG 保存する共通関数。

	Parameters
	----------
	img : np.ndarray
		DICOM から取り出した pixel_array.
	save_dir : Path
		保存先ディレクトリ（患者IDやシリーズ単位など）.
	basename : str
		ファイル名のベース（拡張子やフレーム番号はここで付与する）.
	"""

	save_dir.mkdir(parents=True, exist_ok=True)

	if img.ndim == 2:
		save_path = save_dir / f"{basename}.png"
		Image.fromarray(img).save(save_path)
		print(f"  Saved: {save_path}")
	else:
		# (T, H, W) など複数フレーム想定
		for i in range(img.shape[0]):
			save_path = save_dir / f"{basename}_frame{i:05d}.png"
			Image.fromarray(img[i]).save(save_path)
		print(f"  Saved {img.shape[0]} frames under: {save_dir}")


def save_dicom_array_sequential(img: np.ndarray, save_dir: Path, start_index: int = 1) -> int:
	"""連番 (0001, 0002, ...) で PNG を保存する関数。

	HT での出力仕様に合わせて使用する。

	Parameters
	----------
	img : np.ndarray
		DICOM から取り出した pixel_array.
	save_dir : Path
		保存先ディレクトリ.
	start_index : int
		保存を開始する連番 (1 始まり想定).

	Returns
	-------
	int
		次に使うべき連番 (最後に保存した番号 + 1).
	"""

	save_dir.mkdir(parents=True, exist_ok=True)
	idx = start_index

	if img.ndim == 2:
		fname = f"{idx:04d}.png"
		Image.fromarray(img).save(save_dir / fname)
		print(f"  Saved: {save_dir / fname}")
		idx += 1
	else:
		# (T, H, W) など複数フレーム想定
		for i in range(img.shape[0]):
			fname = f"{idx:04d}.png"
			Image.fromarray(img[i]).save(save_dir / fname)
			idx += 1
		print(f"  Saved {idx - start_index} frames under: {save_dir}")

	return idx


def save_dicom_array_sequential_tiff(img: np.ndarray, save_dir: Path, start_index: int = 1) -> int:
	"""連番 (0001, 0002, ...) で TIFF を保存する関数。

	PNG 保存用関数 ``save_dicom_array_sequential`` と同じ仕様で、
	拡張子のみ .tif にしたバージョン。

	Parameters
	----------
	img : np.ndarray
		DICOM から取り出した pixel_array.
	save_dir : Path
		保存先ディレクトリ.
	start_index : int
		保存を開始する連番 (1 始まり想定).

	Returns
	-------
	int
		次に使うべき連番 (最後に保存した番号 + 1).
	"""

	save_dir.mkdir(parents=True, exist_ok=True)
	idx = start_index

	if img.ndim == 2:
		fname = f"{idx:04d}.tif"
		Image.fromarray(img).save(save_dir / fname)
		print(f"  Saved: {save_dir / fname}")
		idx += 1
	else:
		# (T, H, W) など複数フレーム想定
		for i in range(img.shape[0]):
			fname = f"{idx:04d}.tif"
			Image.fromarray(img[i]).save(save_dir / fname)
			idx += 1
		print(f"  Saved {idx - start_index} frames under: {save_dir}")

	return idx


def save_dicom_metadata(ds: pydicom.dataset.Dataset, save_dir: Path, dicom_path: Path) -> None:
	"""DICOM のメタデータを JSON 形式で保存する共通関数。

	画像 (pixel_array) は扱わず、ヘッダ要素を文字列として保存する。
	同じディレクトリに複数 DICOM が存在するケースもあるため、
	ファイルごとに <dicomファイル名>.json を作成する。
	"""

	save_dir.mkdir(parents=True, exist_ok=True)
	meta: dict[str, str] = {}

	for elem in ds:
		# シーケンスは深くなりすぎるのでひとまずスキップ
		if getattr(elem, "VR", None) == "SQ":
			continue

		# 画像本体はメタデータとしては不要なので PixelData は除外する
		if getattr(elem, "keyword", "") == "PixelData":
			continue

		key = getattr(elem, "keyword", "") or getattr(elem, "name", "") or str(getattr(elem, "tag", ""))
		try:
			value = elem.value
			if isinstance(value, (bytes, bytearray)):
				try:
					value_str = value.decode("utf-8", errors="ignore")
				except Exception:
					value_str = repr(value)
			else:
				value_str = str(value)
		except Exception:
			value_str = repr(getattr(elem, "value", ""))

		meta[key] = value_str

	meta_path = save_dir / f"{dicom_path.stem}.json"
	with meta_path.open("w", encoding="utf-8") as f:
		json.dump(meta, f, ensure_ascii=False, indent=2)

	print(f"  Saved metadata: {meta_path}")


# デフォルトでは PNG で保存する関数を利用する
SAVE_SEQUENTIAL_FUNC = save_dicom_array_sequential
SAVE_MODE = "image"  # "image" or "meta"


_TARGETS = ["HT", "KH", "HK", "KY", "KN", "MN", "SH", "AC", "NT", "NTH", "WJ"]

# HT は今回エラー調査対象の症例だけに絞って走査する
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


_EXCLUDE_SUFFIXES = {
	".png",
	".jpg",
	".jpeg",
	".tif",
	".tiff",
	".json",
	".csv",
	".tsv",
	".txt",
	".md",
	".log",
	".nii",
	".gz",
}


def _should_check_file(p: Path) -> bool:
	if not p.is_file() or p.name.startswith("."):
		return False
	# DICOM は拡張子なし / .dcm などが混在するので、除外リスト方式で雑に弾く
	if p.suffix.lower() in _EXCLUDE_SUFFIXES:
		return False
	return True


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
		return


def _iter_ivus_roots(target: str, patient_dir: Path) -> list[Path]:
	"""患者ディレクトリ配下から 'IVUS' を含むディレクトリを探索する。

	HT は例外的に 'IVUS or OCT files' を IVUS ルート候補として扱う。
	OCT / OFDI はスキップする。
	"""
	roots: list[Path] = []

	if target == "HT":
		root = patient_dir / "IVUS or OCT files"
		if root.exists() and root.is_dir():
			return [root]
		return []

	queue: list[Path] = [patient_dir]
	seen: set[Path] = set()
	while queue:
		current = queue.pop(0)
		if current in seen:
			continue
		seen.add(current)

		name_upper = current.name.upper()
		# OCT / OFDI ディレクトリは丸ごとスキップ
		if "OCT" in name_upper or "OFDI" in name_upper:
			continue

		if "IVUS" in name_upper:
			roots.append(current)

		try:
			subdirs = [
				p for p in sorted(current.iterdir())
				if p.is_dir() and not p.name.startswith(".")
			]
		except Exception:
			continue
		queue.extend(subdirs)

	# 重複除去（Path の順序保持）
	unique: list[Path] = []
	seen2: set[Path] = set()
	for r in roots:
		if r not in seen2:
			unique.append(r)
			seen2.add(r)

	# 入れ子の IVUS ディレクトリがある場合、上位(浅い)のみ残す
	unique_sorted = sorted(unique, key=lambda p: len(p.parts))
	filtered: list[Path] = []
	for r in unique_sorted:
		is_child = False
		for kept in filtered:
			try:
				_ = r.relative_to(kept)
				is_child = True
				break
			except ValueError:
				continue
		if not is_child:
			filtered.append(r)

	return filtered


def scan_target(target: str, ew: ErrorWriter) -> None:
	src_dir = BASE_DATA_DIR / target
	if not src_dir.exists():
		ew.write(str(src_dir), "SRC_NOT_FOUND")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith("."):
			continue

		if target == "HT" and patient_dir.name not in _HT_PATIENTS_TO_SCAN:
			continue

		ivus_roots = _iter_ivus_roots(target, patient_dir)
		if not ivus_roots:
			ew.write(str(patient_dir), "NO_IVUS_DIR")
			continue

		for ivus_root in ivus_roots:
			# HT は "IVUS or OCT files" までで止め、直下ファイルのみを DICOM 候補にする
			# それ以外は従来どおりルート配下を再帰走査
			try:
				if target == "HT":
					files = [p for p in sorted(ivus_root.iterdir()) if _should_check_file(p)]
				else:
					files = [p for p in ivus_root.rglob("*") if _should_check_file(p)]
			except Exception:
				ew.write(str(ivus_root), "CANNOT_TRAVERSE")
				continue

			# 念のためファイル名に OCT / OFDI を含むものは除外
			files = [
				p for p in files
				if "OCT" not in p.name.upper() and "OFDI" not in p.name.upper()
			]

			if not files:
				ew.write(str(ivus_root), "NO_DICOM_FILES")
				continue

			for dicom_file in sorted(files):
				_check_dicom_pixel_array(dicom_file, ew)


def view_ht() -> None:
	"""HT 用ディレクトリ構造に対応した可視化処理。

	入力:
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/HT/HT-***/
		  ├ CT files/
		  ├ IVUS or OCT files/
		  │   ├ Run1/, Run2/, ...         (パターン1)
		  │   └ または 00000001, ...      (パターン2: ファイルが直接並ぶ)
		  └ angio files/

	出力:
		/workspace/data/preprocessed/IVUS/HT/HT-***/<Run名 or 00000001>/0001.png ...
	"""

	src_dir = BASE_DATA_DIR / "HT"
	out_root = CURRENT_SAVE_DIR / "HT"
	print(f"[HT] source dir: {src_dir}")
	print(f"[HT] output root: {out_root}")

	if not src_dir.exists():
		print(f"HT source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):  # .DS_Store などを除外
			continue

		patient_id = patient_dir.name  # 例: HT-001
		ivus_root = patient_dir / "IVUS or OCT files"

		if not ivus_root.exists():
			print(f"  [SKIP] IVUS or OCT files not found for {patient_id}: {ivus_root}")
			continue

		print(f"Processing patient: {patient_id}")

		entries = sorted(ivus_root.iterdir())
		run_dirs = [p for p in entries if p.is_dir() and not p.name.startswith('.')]
		files_direct = [p for p in entries if p.is_file() and not p.name.startswith('.')]

		# パターン1: Run1/, Run2/ ... のようにサブディレクトリがある場合
		if run_dirs:
			for run_dir in run_dirs:
				run_name = run_dir.name  # 例: Run1
				print(f"  Run dir: {run_name}")

				# Run ディレクトリ直下のすべてのファイルを DICOM とみなして処理
				# （拡張子なし 00000001 などにも対応するため glob("*.dcm") は使わない）
				dicom_files = [
					p for p in sorted(run_dir.iterdir())
					if p.is_file() and not p.name.startswith('.')
				]

				if not dicom_files:
					print(f"    [WARN] No files found in {run_dir}")
					continue

				save_dir = out_root / patient_id / run_name
				idx = 1

				for dicom_file in dicom_files:
					print(f"		Processing DICOM: {dicom_file}")
					try:
						ds = pydicom.dcmread(dicom_file, force=True)
					except Exception as e:
						print(f"		  Error reading {dicom_file}: {e}")
						continue

					if SAVE_MODE == "meta":
						save_dicom_metadata(ds, save_dir, dicom_file)
						continue

					try:
						img = ds.pixel_array
					except Exception as e:
						print(f"		  Error getting pixel_array from {dicom_file}: {e}")
						continue

					print(
						f"		  shape: {img.shape}, dtype: {img.dtype}, "
						f"min: {np.min(img)}, max: {np.max(img)}"
					)

					idx = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)

		# パターン2: IVUS or OCT files 直下に 00000001, 00000002 ... のような DICOM ファイルが並ぶ場合
		elif files_direct:
			for dicom_file in files_direct:
				series_name = dicom_file.name  # 例: 00000001
				print(f"  Series file: {series_name}")

				save_dir = out_root / patient_id / series_name
				idx = 1

				try:
					ds = pydicom.dcmread(dicom_file, force=True)
				except Exception as e:
					print(f"    Error reading {dicom_file}: {e}")
					continue

				if SAVE_MODE == "meta":
					save_dicom_metadata(ds, save_dir, dicom_file)
					continue

				try:
					img = ds.pixel_array
				except Exception as e:
					print(f"    Error getting pixel_array from {dicom_file}: {e}")
					continue

				print(
					f"    shape: {img.shape}, dtype: {img.dtype}, "
					f"min: {np.min(img)}, max: {np.max(img)}"
				)

				_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)

		else:
			print(f"  [WARN] No valid entries under: {ivus_root}")


def view_kh() -> None:
	"""KH 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/KH/
		  ├ KH-IVUS-1(KH2-20)/
		  ├ KH-IVUS-2(KH21-35)/
		  └ KH-IVUS-3(KH36-42)/
		        └ DATA.KH-36/20220215103845/20220215/*.dcm

	出力:
		/workspace/data/preprocessed/IVUS/KH/KH-036/<元ファイル名>/0001.png ...
	"""

	src_dir = BASE_DATA_DIR / "KH"
	out_root = CURRENT_SAVE_DIR / "KH"
	print(f"[KH] source dir: {src_dir}")
	print(f"[KH] output root: {out_root}")

	if not src_dir.exists():
		print(f"KH source dir not found: {src_dir}")
		return

	# KH 配下の KH-IVUS-* ディレクトリのみ対象にする (KH-OCT などは除外)
	for session_dir in sorted(src_dir.iterdir()):
		if not session_dir.is_dir() or session_dir.name.startswith('.'):
			continue
		if not session_dir.name.startswith("KH-IVUS"):
			continue

		print(f"Session dir: {session_dir.name}")

		for data_dir in sorted(session_dir.iterdir()):
			if not data_dir.is_dir() or data_dir.name.startswith('.'):
				continue
			if not data_dir.name.startswith("DATA.KH-"):
				continue

			# DATA.KH-36 -> KH-036 のように正規化
			data_name = data_dir.name  # 例: DATA.KH-36
			try:
				_, kh_part = data_name.split("DATA.", maxsplit=1)  # KH-36
				prefix, num_str = kh_part.split("-", maxsplit=1)   # ("KH", "36")
				num = int(num_str)
				patient_id = f"{prefix}-{num:03d}"              # KH-036
			except Exception:
				print(f"  [WARN] Unexpected DATA dir name: {data_name}")
				continue

			print(f"  DATA dir: {data_name} -> patient_id: {patient_id}")
			patient_out_root = out_root / patient_id

			# DATA.KH-36/ 以下の階層をたどり、最終的に DICOM ファイルが置かれている
			# ディレクトリを見つけて処理する
			for level1 in sorted(data_dir.iterdir()):
				if not level1.is_dir() or level1.name.startswith('.'):
					continue

				# 例: 20220215103845
				subdirs = [
					p for p in sorted(level1.iterdir())
					if p.is_dir() and not p.name.startswith('.')
				]

				leaf_dirs: list[Path]
				if subdirs:
					# 例: 20220215103845/20220215
					leaf_dirs = subdirs
				else:
					# level1 直下にファイルがあるパターンも一応サポート
					leaf_dirs = [level1]

				for leaf_dir in leaf_dirs:
					print(f"    Leaf dir: {leaf_dir}")
					# leaf_dir 直下の .dcm を処理
					dicom_files = [
						p for p in sorted(leaf_dir.iterdir())
						if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == ".dcm"
					]

					if not dicom_files:
						print(f"      [WARN] No DICOM files in {leaf_dir}")
						continue

					for dicom_file in dicom_files:
						file_stem = dicom_file.stem
						save_dir = patient_out_root / file_stem
						idx = 1

						print(f"      Processing DICOM: {dicom_file}")
						try:
							ds = pydicom.dcmread(dicom_file, force=True)
							img = ds.pixel_array
						except Exception as e:
							print(f"        Error reading {dicom_file}: {e}")
							continue

						print(
							f"        shape: {img.shape}, dtype: {img.dtype}, "
							f"min: {np.min(img)}, max: {np.max(img)}"
						)

						_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def _normalize_hk_patient_id(patient_raw: str) -> str:
	"""HK の患者ディレクトリ名を出力用 ID (HK_003, HK_008-1 など) に正規化する。"""
	try:
		prefix, rest = patient_raw.split("-", maxsplit=1)
	except ValueError:
		return patient_raw

	parts = rest.split("-")
	try:
		main_num = int(parts[0])
	except ValueError:
		return patient_raw

	if len(parts) == 1:
		return f"{prefix}_{main_num:03d}"
	else:
		sub = "-".join(parts[1:])
		return f"{prefix}_{main_num:03d}-{sub}"


def _process_hk_patient_dir(patient_dir: Path, out_root: Path) -> None:
	patient_raw = patient_dir.name
	patient_id = _normalize_hk_patient_id(patient_raw)
	patient_out_root = out_root / patient_id

	print(f"Processing patient: {patient_raw} -> {patient_id}")

	# 例: HK-3/HK-3(IVUS)/Unnamed/...
	session_dirs = [
		p for p in sorted(patient_dir.iterdir())
		if p.is_dir() and not p.name.startswith('.') and "IVUS" in p.name.upper()
	]

	if not session_dirs:
		print(f"  [WARN] No IVUS session dir under: {patient_dir}")
		return

	for session_dir in session_dirs:
		print(f"  Session dir: {session_dir}")

		# Unnamed (または Unnamed* ) ディレクトリを探す
		unnamed_dirs = [
			p for p in sorted(session_dir.iterdir())
			if p.is_dir() and not p.name.startswith('.') and p.name.lower().startswith("unnamed")
		]

		if not unnamed_dirs:
			print(f"    [WARN] No 'Unnamed' dir under: {session_dir}")
			continue

		for unnamed_dir in unnamed_dirs:
			print(f"    Unnamed dir: {unnamed_dir}")

			# Unnamed 配下のディレクトリごとにシリーズとして扱う
			series_dirs = [
				p for p in sorted(unnamed_dir.iterdir())
				if p.is_dir() and not p.name.startswith('.')
			]

			if not series_dirs:
				print(f"      [WARN] No series dirs under: {unnamed_dir}")
				continue

			for series_dir in series_dirs:
				series_name = series_dir.name
				print(f"      Series dir: {series_dir}")

				# シリーズ配下のファイルを DICOM とみなす
				dicom_files = [
					p for p in sorted(series_dir.iterdir())
					if p.is_file() and not p.name.startswith('.')
				]

				if not dicom_files:
					print(f"        [WARN] No DICOM-like files under: {series_dir}")
					continue

				for dicom_file in dicom_files:
					# ファイル名ごとにディレクトリを分けて保存
					if dicom_file.suffix.lower() == ".dcm":
						file_key = dicom_file.stem
					else:
						file_key = dicom_file.name

					save_dir = patient_out_root / series_name / file_key
					idx = 1

					print(f"        Processing DICOM: {dicom_file} -> series {series_name}, file {file_key}")
					try:
						ds = pydicom.dcmread(dicom_file, force=True)
					except Exception as e:
						print(f"          Error reading {dicom_file}: {e}")
						continue

					if SAVE_MODE == "meta":
						save_dicom_metadata(ds, save_dir, dicom_file)
						continue

					try:
						img = ds.pixel_array
					except Exception as e:
						print(f"          Error getting pixel_array from {dicom_file}: {e}")
						continue

					print(
						f"          shape: {img.shape}, dtype: {img.dtype}, "
						f"min: {np.min(img)}, max: {np.max(img)}"
					)

					_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def _normalize_ky_patient_id(patient_raw: str) -> str:
	"""KY の患者ディレクトリ名を KY-015 のように 3 桁ゼロ埋めで正規化する。"""
	try:
		prefix, num_str = patient_raw.split("-", maxsplit=1)
		num = int(num_str)
	except Exception:
		return patient_raw

	return f"{prefix}-{num:03d}"


def _normalize_ky_group_name(name: str) -> str:
	"""KY-01 配下の ① などのディレクトリ名を 1,2,... に正規化する。

	数字が含まれていれば最初の数字を返し、
	含まれない場合は circled number などをマップして返す。
	どれにも当てはまらなければ元の名前を返す。
	"""

	circled_map = {
		"①": "1",
		"②": "2",
		"③": "3",
		"④": "4",
		"⑤": "5",
		"⑥": "6",
		"⑦": "7",
		"⑧": "8",
		"⑨": "9",
	}

	for ch in name:
		if ch.isdigit():
			return ch
		if ch in circled_map:
			return circled_map[ch]

	return name


def _normalize_nt_patient_id(patient_raw: str) -> str:
	"""NT の患者ディレクトリ名を NT-017 のように 3 桁ゼロ埋めで正規化する。"""
	try:
		prefix, num_str = patient_raw.split("-", maxsplit=1)
		num = int(num_str)
	except Exception:
		return patient_raw

	return f"{prefix}-{num:03d}"


def _normalize_nth_patient_id(patient_raw: str) -> str:
	"""NTH の患者ディレクトリ名を NTH-096 のように 3 桁ゼロ埋めで正規化する。

	ディレクトリ名は "NTH-096 IVUS" のように末尾に " IVUS" が付くことを想定しているため、
	空白以降は無視して扱う。
	"""

	base = patient_raw.split()[0]
	try:
		prefix, num_str = base.split("-", maxsplit=1)
		num = int(num_str)
	except Exception:
		return base

	return f"{prefix}-{num:03d}"


def _normalize_wj_patient_id(patient_raw: str) -> str:
	"""WJ の患者ディレクトリ名を WJ-008 のように 3 桁ゼロ埋めで正規化する。"""
	try:
		prefix, num_str = patient_raw.split("-", maxsplit=1)
		num = int(num_str)
	except Exception:
		return patient_raw

	return f"{prefix}-{num:03d}"


def view_kn() -> None:
	"""KN 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/KN/
		  └ KN-18/
		       └ KN-18 IVUS 2025.01.15/
		            └ DICOM/00001IMG, 00002IMG, ...

	出力:
		/workspace/data/preprocessed/IVUS/KN/KN-018/<ファイル名>/0001.png ...
	"""

	src_dir = BASE_DATA_DIR / "KN"
	out_root = CURRENT_SAVE_DIR / "KN"
	print(f"[KN] source dir: {src_dir}")
	print(f"[KN] output root: {out_root}")

	if not src_dir.exists():
		print(f"KN source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("KN-"):
			continue

		# KN-18 -> KN-018 に正規化
		patient_raw = patient_dir.name  # 例: KN-18
		try:
			prefix, num_str = patient_raw.split("-", maxsplit=1)
			num = int(num_str)
			patient_id = f"{prefix}-{num:03d}"  # KN-018
		except Exception:
			print(f"  [WARN] Unexpected patient dir name: {patient_raw}")
			continue

		print(f"Processing patient: {patient_raw} -> {patient_id}")
		patient_out_root = out_root / patient_id

		# まず "<KN-18> IVUS" で始まるディレクトリを探す
		ivus_dirs = [
			p for p in sorted(patient_dir.iterdir())
			if p.is_dir() and p.name.startswith(f"{patient_raw} IVUS")
		]

		# KN-1 ～ KN-9 のように患者ディレクトリ名が 1 桁だが、
		# その下の IVUS ディレクトリ名が KN-01 IVUS ... となっているケースにも対応する
		if not ivus_dirs and patient_raw.startswith("KN-"):
			try:
				prefix, num_str = patient_raw.split("-", maxsplit=1)
				num = int(num_str)
				alt_name = f"{prefix}-{num:02d}"
			except Exception:
				alt_name = None

			if alt_name is not None:
				ivus_dirs = [
					p for p in sorted(patient_dir.iterdir())
					if p.is_dir() and p.name.startswith(f"{alt_name} IVUS")
				]

		if ivus_dirs:
			ivus_root = ivus_dirs[0]
		else:
			# フォールバック: 直下の DICOM ディレクトリ
			ivus_root = patient_dir / "DICOM"
			if not ivus_root.exists():
				print(f"  [SKIP] No IVUS dir or DICOM dir for {patient_raw}")
				continue

		print(f"  IVUS root: {ivus_root}")

		# IVUS ルート直下の DICOM ディレクトリを決める
		dicom_root = ivus_root / "DICOM"
		if not dicom_root.exists():
			# すでに DICOM 直下だった場合も一応サポート
			dicom_root = ivus_root

		if not dicom_root.exists():
			print(f"  [SKIP] DICOM dir not found: {dicom_root}")
			continue

		print(f"  DICOM root: {dicom_root}")

		# DICOM 直下のファイル (00001IMG など、拡張子なしも含む) を処理
		dicom_files = [
			p for p in sorted(dicom_root.iterdir())
			if p.is_file() and not p.name.startswith('.')
		]

		if not dicom_files:
			print(f"  [WARN] No DICOM-like files under: {dicom_root}")
			continue

		for dicom_file in dicom_files:
			file_name = dicom_file.name  # 例: 00001IMG
			save_dir = patient_out_root / file_name
			idx = 1

			print(f"    Processing DICOM: {dicom_file}")
			try:
				ds = pydicom.dcmread(dicom_file, force=True)
			except Exception as e:
				print(f"Error reading {dicom_file}")
				print(f"type={type(e)}")
				print(f"repr={repr(e)}")
				continue

			if SAVE_MODE == "meta":
				save_dicom_metadata(ds, save_dir, dicom_file)
				continue

			print("TransferSyntaxUID:", getattr(ds.file_meta, "TransferSyntaxUID", "N/A"))
			print("Modality:", getattr(ds, "Modality", "N/A"))
			print("Rows:", getattr(ds, "Rows", "N/A"))
			print("Columns:", getattr(ds, "Columns", "N/A"))
			print("BitsAllocated:", getattr(ds, "BitsAllocated", "N/A"))
			print("SamplesPerPixel:", getattr(ds, "SamplesPerPixel", "N/A"))
			print("PhotometricInterpretation:", getattr(ds, "PhotometricInterpretation", "N/A"))

			try:
				img = ds.pixel_array
			except Exception as e:
				print(f"Error getting pixel_array from {dicom_file}")
				print(f"type={type(e)}")
				print(f"repr={repr(e)}")
				continue

			print(
				f"      shape: {img.shape}, dtype: {img.dtype}, "
				f"min: {np.min(img)}, max: {np.max(img)}"
			)

			_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_mn() -> None:
	"""MN 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/MN/
		  └ MN-22/
		       ├ IVUS0001.dcm
		       ├ IVUS0002.dcm
		       ├ MN-22CT/
		       └ MN-22PCI/

	出力:
		/workspace/data/preprocessed/IVUS/MN/MN-022/<ファイル名>/0001.png ...
	"""

	src_dir = BASE_DATA_DIR / "MN"
	out_root = CURRENT_SAVE_DIR / "MN"
	print(f"[MN] source dir: {src_dir}")
	print(f"[MN] output root: {out_root}")

	if not src_dir.exists():
		print(f"MN source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("MN-"):
			continue

		# MN-22 -> MN-022 に正規化
		patient_raw = patient_dir.name
		try:
			prefix, num_str = patient_raw.split("-", maxsplit=1)
			num = int(num_str)
			patient_id = f"{prefix}-{num:03d}"  # MN-022
		except Exception:
			print(f"  [WARN] Unexpected patient dir name: {patient_raw}")
			continue

		print(f"Processing patient: {patient_raw} -> {patient_id}")
		patient_out_root = out_root / patient_id

		# 患者ディレクトリ直下の .dcm ファイルのみを IVUS として扱う
		dicom_files = [
			p for p in sorted(patient_dir.iterdir())
			if p.is_file() and not p.name.startswith('.') and p.suffix.lower() == ".dcm"
		]

		if not dicom_files:
			print(f"  [WARN] No DICOM files directly under: {patient_dir}")
			continue

		for dicom_file in dicom_files:
			file_stem = dicom_file.stem  # 例: IVUS0001
			save_dir = patient_out_root / file_stem
			idx = 1

			print(f"    Processing DICOM: {dicom_file}")
			try:
				ds = pydicom.dcmread(dicom_file, force=True)
			except Exception as e:
				print(f"	      Error reading {dicom_file}: {e}")
				continue

			if SAVE_MODE == "meta":
				save_dicom_metadata(ds, save_dir, dicom_file)
				continue

			try:
				img = ds.pixel_array
			except Exception as e:
				print(f"	      Error getting pixel_array from {dicom_file}: {e}")
				continue

			print(
				f"	      shape: {img.shape}, dtype: {img.dtype}, "
				f"min: {np.min(img)}, max: {np.max(img)}"
			)

			_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_sh() -> None:
	"""SH 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/SH/
		  └ SH-4/
		       └ IVUS/DICOM/00000000/00000000, 00000001, ...

	出力:
		/workspace/data/preprocessed/IVUS/SH/SH-004/<ファイル名>/0001.png ...
	"""

	src_dir = BASE_DATA_DIR / "SH"
	out_root = CURRENT_SAVE_DIR / "SH"
	print(f"[SH] source dir: {src_dir}")
	print(f"[SH] output root: {out_root}")

	if not src_dir.exists():
		print(f"SH source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("SH-"):
			continue

		# SH-4 -> SH-004 に正規化
		patient_raw = patient_dir.name
		try:
			prefix, num_str = patient_raw.split("-", maxsplit=1)
			num = int(num_str)
			patient_id = f"{prefix}-{num:03d}"  # SH-004
		except Exception:
			print(f"  [WARN] Unexpected patient dir name: {patient_raw}")
			continue

		print(f"Processing patient: {patient_raw} -> {patient_id}")
		patient_out_root = out_root / patient_id

		ivus_root = patient_dir / "IVUS" / "DICOM"
		if not ivus_root.exists():
			print(f"  [SKIP] IVUS/DICOM not found for {patient_raw}: {ivus_root}")
			continue

		print(f"  IVUS DICOM root: {ivus_root}")

		# 例: IVUS/DICOM/00000000/ 以下にファイルが並ぶ想定
		for series_dir in sorted(ivus_root.iterdir()):
			if not series_dir.is_dir() or series_dir.name.startswith('.'):
				continue

			series_name = series_dir.name  # 例: 00000000
			print(f"  Series dir: {series_dir}")

			# series_dir 直下のファイル (00000000 など) を DICOM として処理
			dicom_files = [
				p for p in sorted(series_dir.iterdir())
				if p.is_file() and not p.name.startswith('.')
			]

			if not dicom_files:
				print(f"    [WARN] No DICOM-like files under: {series_dir}")
				continue

			for dicom_file in dicom_files:
				file_name = dicom_file.name  # 例: 00000000
				save_dir = patient_out_root / file_name
				idx = 1

				print(f"    Processing DICOM: {dicom_file}")
				try:
					ds = pydicom.dcmread(dicom_file, force=True)
				except Exception as e:
					print(f"	      Error reading {dicom_file}: {e}")
					continue

				if SAVE_MODE == "meta":
					save_dicom_metadata(ds, save_dir, dicom_file)
					continue

				try:
					img = ds.pixel_array
				except Exception as e:
					print(f"	      Error getting pixel_array from {dicom_file}: {e}")
					continue

				print(
					f"	      shape: {img.shape}, dtype: {img.dtype}, "
					f"min: {np.min(img)}, max: {np.max(img)}"
				)

				_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_ac() -> None:
	"""AC 用ディレクトリ構造に対応した可視化処理。

	想定構造の例:
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/AC/AC-2/AC-2(00340950)/2025.11.11 IVUS/20210519180729/20210519/00000001
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/AC/AC-7/IVUS
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/AC/AC-12/IVUS/VISICUBE/DATA/20220217151441/20220217/00000001

	出力:
		/workspace/data/preprocessed/IVUS/AC/AC-002/<系列名>/0001.png ...

	- OFDI や OCT は別データのためスキップする。
	- パス中に "IVUS" を含むディレクトリを IVUS ルートとして扱い、
	  その配下で DICOM ファイルを含む最下層ディレクトリごとに出力する。
	- ただし、IVUS 直下に DICOM ファイルが並んでいる場合は、
	  各ファイル名ごとにディレクトリを分けて保存する。
	"""

	src_dir = BASE_DATA_DIR / "AC"
	out_root = CURRENT_SAVE_DIR / "AC"
	print(f"[AC] source dir: {src_dir}")
	print(f"[AC] output root: {out_root}")

	if not src_dir.exists():
		print(f"AC source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("AC-"):
			continue

		# AC-2 -> AC-002 に正規化
		patient_raw = patient_dir.name
		try:
			prefix, num_str = patient_raw.split("-", maxsplit=1)
			num = int(num_str)
			patient_id = f"{prefix}-{num:03d}"  # AC-002
		except Exception:
			print(f"  [WARN] Unexpected patient dir name: {patient_raw}")
			continue

		print(f"Processing patient: {patient_raw} -> {patient_id}")
		patient_out_root = out_root / patient_id

		# 患者ディレクトリ以下で "IVUS" を含むディレクトリを IVUS ルートとして列挙
		ivus_roots: list[Path] = []
		queue: list[Path] = [patient_dir]

		while queue:
			current = queue.pop(0)
			for p in sorted(current.iterdir()):
				if not p.is_dir() or p.name.startswith('.'):
					continue

				name_upper = p.name.upper()
				# OFDI / OCT ディレクトリはスキップ
				if "OCT" in name_upper or "OFDI" in name_upper:
					continue

				if "IVUS" in name_upper:
					ivus_roots.append(p)
					queue.append(p)
				else:
					queue.append(p)

		if not ivus_roots:
			print(f"  [WARN] No IVUS dir found for {patient_raw}")
			continue

		for ivus_root in ivus_roots:
			print(f"  IVUS root: {ivus_root}")

			# ivus_root 以下で DICOM らしきファイルを含むディレクトリを列挙
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
				print(f"    [WARN] No DICOM-like files under: {ivus_root}")
				continue

			# 最下層（より深い）ディレクトリのみ残す
			candidate_dirs = sorted(set(candidate_dirs), key=lambda p: len(p.parts), reverse=True)
			leaf_dirs: list[Path] = []

			for d in candidate_dirs:
				is_ancestor = False
				for s in leaf_dirs:
					try:
						_ = s.relative_to(d)
						is_ancestor = True
						break
					except ValueError:
						continue

				if not is_ancestor:
					leaf_dirs.append(d)

			for leaf_dir in sorted(leaf_dirs):
				print(f"    Leaf dir: {leaf_dir}")
				try:
					files = [
						p for p in sorted(leaf_dir.iterdir())
						if p.is_file() and not p.name.startswith('.')
					]
				except Exception:
					print(f"      [WARN] Cannot list files in {leaf_dir}")
					continue

				if not files:
					print(f"      [WARN] No files in leaf dir: {leaf_dir}")
					continue

				# パターン1: IVUS 直下に DICOM ファイルが並んでいる場合
				if leaf_dir == ivus_root:
					for dicom_file in files:
						name_upper = dicom_file.name.upper()
						# 念のため OFDI / OCT を含むファイル名はスキップ
						if "OCT" in name_upper or "OFDI" in name_upper:
							continue

						if dicom_file.suffix.lower() == ".dcm":
							series_name = dicom_file.stem
						else:
							series_name = dicom_file.name

						save_dir = patient_out_root / series_name
						idx = 1

						print(f"      Processing DICOM (file-level): {dicom_file} -> series {series_name}")
						try:
							ds = pydicom.dcmread(dicom_file, force=True)
						except Exception as e:
							print(f"        Error reading {dicom_file}: {e}")
							continue

						if SAVE_MODE == "meta":
							save_dicom_metadata(ds, save_dir, dicom_file)
							continue

						try:
							img = ds.pixel_array
						except Exception as e:
							print(f"        Error getting pixel_array from {dicom_file}: {e}")
							continue

						print(
							f"        shape: {img.shape}, dtype: {img.dtype}, "
							f"min: {np.min(img)}, max: {np.max(img)}"
						)

						_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)

				# パターン2: 日付 IVUS / VISICUBE など配下の最下層ディレクトリごとに処理
				else:
					# leaf_dir (例: 20210519) を 1 series としつつ、
					# その配下では DICOM ファイルごとに保存先ディレクトリを分ける。
					# (複数ファイルを同一 save_dir に start_index=1 で保存すると上書きが起きるため)
					series_name = leaf_dir.name

					for dicom_file in files:
						# ファイル名ごとにディレクトリを分けて保存
						if dicom_file.suffix.lower() == ".dcm":
							file_key = dicom_file.stem
						else:
							file_key = dicom_file.name

						# 出力パスに日付ディレクトリ (series_name) は含めない
						save_dir = patient_out_root / file_key
						idx = 1

						print(
							f"      Processing DICOM: {dicom_file} -> (leaf={series_name}) file {file_key}"
						)
						try:
							ds = pydicom.dcmread(dicom_file, force=True)
						except Exception as e:
							print(f"        Error reading {dicom_file}: {e}")
							continue

						if SAVE_MODE == "meta":
							save_dicom_metadata(ds, save_dir, dicom_file)
							continue

						try:
							img = ds.pixel_array
						except Exception as e:
							print(f"        Error getting pixel_array from {dicom_file}: {e}")
							continue

						print(
							f"        shape: {img.shape}, dtype: {img.dtype}, "
							f"min: {np.min(img)}, max: {np.max(img)}"
						)

						_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_hk() -> None:
	"""HK 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/HK/HK-3/HK-3(IVUS)/Unnamed/<シリーズ名>/<DICOM ファイル群>

	出力:
		/workspace/data/preprocessed/IVUS/HK/HK_003/<シリーズ名>/<ファイル名>/0001.png ...

	特殊ケース:
		HK-8-2 の配下に HK-8-1 と HK-8-2 がある場合、それぞれを
		HK_008-1, HK_008-2 として別々に処理する。
	"""

	src_dir = BASE_DATA_DIR / "HK"
	out_root = CURRENT_SAVE_DIR / "HK"
	print(f"[HK] source dir: {src_dir}")
	print(f"[HK] output root: {out_root}")

	if not src_dir.exists():
		print(f"HK source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("HK-"):
			continue

		# 特殊ケース: HK-8-2 の下に HK-8-1 / HK-8-2 などがぶら下がっている場合は、
		# それぞれを個別の症例として処理する。
		if patient_dir.name == "HK-8-2":
			for sub_dir in sorted(patient_dir.iterdir()):
				if not sub_dir.is_dir() or sub_dir.name.startswith('.'):
					continue
				if not sub_dir.name.startswith("HK-"):
					continue

				_process_hk_patient_dir(sub_dir, out_root)
		else:
			_process_hk_patient_dir(patient_dir, out_root)


def view_ky() -> None:
	"""KY 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/KY/KY-15/IVUS
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/KY/KY-01/①/IVUS

	出力:
		/workspace/data/preprocessed/IVUS/KY/KY-015/<ファイル名>/0001.png ...
		/workspace/data/preprocessed/IVUS/KY/KY-001/1/<ファイル名>/0001.png ...

	- KY-XX 直下に IVUS がある場合は、KY-0XX の配下にファイル名ごとのディレクトリを切って保存。
	- KY-XX/①/IVUS のように複数のディレクトリに IVUS がある場合は、
	  ① を 1, ② を 2 などに正規化したサブディレクトリを掘り、その下にファイル名ごとのディレクトリを作成する。
	- IVUS 以下の中間ディレクトリ名は無視し、DICOM ファイル名のみでディレクトリを分ける。
	"""

	src_dir = BASE_DATA_DIR / "KY"
	out_root = CURRENT_SAVE_DIR / "KY"
	print(f"[KY] source dir: {src_dir}")
	print(f"[KY] output root: {out_root}")

	if not src_dir.exists():
		print(f"KY source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("KY-"):
			continue

		patient_raw = patient_dir.name
		patient_id = _normalize_ky_patient_id(patient_raw)
		patient_out_root = out_root / patient_id

		print(f"Processing patient: {patient_raw} -> {patient_id}")

		ivus_roots: list[tuple[Path, str | None]] = []

		# パターン1: KY-15/IVUS
		ivus_dir_direct = patient_dir / "IVUS"
		if ivus_dir_direct.exists() and ivus_dir_direct.is_dir():
			ivus_roots.append((ivus_dir_direct, None))

		# パターン2: KY-01/①/IVUS, KY-01/②/IVUS, ...
		for sub_dir in sorted(patient_dir.iterdir()):
			if not sub_dir.is_dir() or sub_dir.name.startswith('.'):
				continue
			if sub_dir.name == "IVUS":  # すでに直接パターンで拾っている
				continue

			ivus_under_sub = sub_dir / "IVUS"
			if ivus_under_sub.exists() and ivus_under_sub.is_dir():
				group_name = _normalize_ky_group_name(sub_dir.name)
				ivus_roots.append((ivus_under_sub, group_name))

		if not ivus_roots:
			print(f"  [WARN] No IVUS dir found for {patient_raw}")
			continue

		for ivus_root, group_name in ivus_roots:
			base_out = patient_out_root if group_name is None else patient_out_root / group_name
			print(f"  IVUS root: {ivus_root} (group={group_name})")

			# IVUS ディレクトリ以下を再帰的にたどり、ファイルをすべて DICOM とみなす
			try:
				all_files = [
					p for p in ivus_root.rglob("*")
					if p.is_file() and not p.name.startswith('.')
				]
			except Exception:
				print(f"  [WARN] Cannot traverse under IVUS root: {ivus_root}")
				continue

			if not all_files:
				print(f"  [WARN] No DICOM-like files under: {ivus_root}")
				continue

			for dicom_file in sorted(all_files):
				# ファイル名ごとにディレクトリを分けて保存
				if dicom_file.suffix.lower() == ".dcm":
					file_key = dicom_file.stem
				else:
					file_key = dicom_file.name

				save_dir = base_out / file_key
				idx = 1

				print(f"    Processing DICOM: {dicom_file} -> file {file_key}")
				try:
					ds = pydicom.dcmread(dicom_file, force=True)
				except Exception as e:
					print(f"      Error reading {dicom_file}: {e}")
					continue

				if SAVE_MODE == "meta":
					save_dicom_metadata(ds, save_dir, dicom_file)
					continue

				try:
					img = ds.pixel_array
				except Exception as e:
					print(f"      Error getting pixel_array from {dicom_file}: {e}")
					continue

				print(
					f"      shape: {img.shape}, dtype: {img.dtype}, "
					f"min: {np.min(img)}, max: {np.max(img)}"
				)

				_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_nt() -> None:
	"""NT 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/NT/NT-17/IVUS/DICOM/00000001

	出力:
		/workspace/data/preprocessed/IVUS/NT/NT-017/<ファイル名>/0001.png ...

	- NT-XX/IVUS/DICOM/ 以下を再帰的にたどり、すべてのファイルを DICOM とみなす。
	- 中間ディレクトリ名は無視し、DICOM ファイル名をディレクトリ名として保存する。
	"""

	src_dir = BASE_DATA_DIR / "NT"
	out_root = CURRENT_SAVE_DIR / "NT"
	print(f"[NT] source dir: {src_dir}")
	print(f"[NT] output root: {out_root}")

	if not src_dir.exists():
		print(f"NT source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("NT-"):
			continue

		patient_raw = patient_dir.name
		patient_id = _normalize_nt_patient_id(patient_raw)
		patient_out_root = out_root / patient_id

		print(f"Processing patient: {patient_raw} -> {patient_id}")

		# 基本パターン: NT-17/IVUS/DICOM
		ivus_root = patient_dir / "IVUS" / "DICOM"
		if not ivus_root.exists():
			print(f"  [SKIP] IVUS/DICOM not found for {patient_raw}: {ivus_root}")
			continue

		print(f"  IVUS DICOM root: {ivus_root}")

		try:
			all_files = [
				p for p in ivus_root.rglob("*")
				if p.is_file() and not p.name.startswith('.')
			]
		except Exception:
			print(f"  [WARN] Cannot traverse under IVUS/DICOM root: {ivus_root}")
			continue

		if not all_files:
			print(f"  [WARN] No DICOM-like files under: {ivus_root}")
			continue

		for dicom_file in sorted(all_files):
			# ファイル名ごとにディレクトリを分けて保存
			if dicom_file.suffix.lower() == ".dcm":
				file_key = dicom_file.stem
			else:
				file_key = dicom_file.name

			save_dir = patient_out_root / file_key
			idx = 1

			print(f"    Processing DICOM: {dicom_file} -> file {file_key}")
			try:
				ds = pydicom.dcmread(dicom_file, force=True)
			except Exception as e:
				print(f"      Error reading {dicom_file}: {e}")
				continue

			if SAVE_MODE == "meta":
				save_dicom_metadata(ds, save_dir, dicom_file)
				continue

			try:
				img = ds.pixel_array
			except Exception as e:
				print(f"      Error getting pixel_array from {dicom_file}: {e}")
				continue

			print(
				f"      shape: {img.shape}, dtype: {img.dtype}, "
				f"min: {np.min(img)}, max: {np.max(img)}"
			)

			_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_nth() -> None:
	"""NTH 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/NTH/NTH-096 IVUS/XA000001

	出力:
		/workspace/data/preprocessed/IVUS/NTH/NTH-096/<ファイル名>/0001.png ...

	- NTH 直下の "NTH-*** IVUS" ディレクトリを患者単位とみなし、
	  その直下にある XA から始まるファイルを DICOM として扱う。
	- 中間ディレクトリは存在しない前提で、ファイル名をそのままディレクトリ名にして保存する。
	"""

	src_dir = BASE_DATA_DIR / "NTH"
	out_root = CURRENT_SAVE_DIR / "NTH"
	print(f"[NTH] source dir: {src_dir}")
	print(f"[NTH] output root: {out_root}")

	if not src_dir.exists():
		print(f"NTH source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("NTH-"):
			continue

		patient_raw = patient_dir.name  # 例: NTH-096 IVUS
		patient_id = _normalize_nth_patient_id(patient_raw)
		patient_out_root = out_root / patient_id

		print(f"Processing patient: {patient_raw} -> {patient_id}")

		# 直下の XA* ファイルを DICOM として扱う
		try:
			files = [
				p for p in sorted(patient_dir.iterdir())
				if p.is_file()
				and not p.name.startswith('.')
				and p.name.upper().startswith("XA")
			]
		except Exception:
			print(f"  [WARN] Cannot list files under: {patient_dir}")
			continue

		if not files:
			print(f"  [WARN] No XA* DICOM-like files under: {patient_dir}")
			continue

		for dicom_file in files:
			# ファイル名ごとにディレクトリを分けて保存
			if dicom_file.suffix.lower() == ".dcm":
				file_key = dicom_file.stem
			else:
				file_key = dicom_file.name

			save_dir = patient_out_root / file_key
			idx = 1

			print(f"    Processing DICOM: {dicom_file} -> file {file_key}")
			try:
				ds = pydicom.dcmread(dicom_file, force=True)
			except Exception as e:
				print(f"      Error reading {dicom_file}: {e}")
				continue

			if SAVE_MODE == "meta":
				save_dicom_metadata(ds, save_dir, dicom_file)
				continue

			try:
				img = ds.pixel_array
			except Exception as e:
				print(f"      Error getting pixel_array from {dicom_file}: {e}")
				continue

			print(
				f"      shape: {img.shape}, dtype: {img.dtype}, "
				f"min: {np.min(img)}, max: {np.max(img)}"
			)

			_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


def view_wj() -> None:
	"""WJ 用ディレクトリ構造に対応した可視化処理。

	想定構造 (例):
		/workspace/data/org_data/IVUS/CCTI-calc-IVUS_OCT/WJ/WJ-8/IVUS

	出力:
		/workspace/data/preprocessed/IVUS/WJ/WJ-008/<ファイル名>/0001.png ...

	- WJ-XX 直下に "IVUS" ディレクトリがある場合のみ対象とし、
	  その中の XA から始まるファイルを DICOM として扱う。
	- IVUS ディレクトリが存在しない症例はスキップする。
	- 中間ディレクトリは想定せず、ファイル名をそのままディレクトリ名にして保存する。
	"""

	src_dir = BASE_DATA_DIR / "WJ"
	out_root = CURRENT_SAVE_DIR / "WJ"
	print(f"[WJ] source dir: {src_dir}")
	print(f"[WJ] output root: {out_root}")

	if not src_dir.exists():
		print(f"WJ source dir not found: {src_dir}")
		return

	for patient_dir in sorted(src_dir.iterdir()):
		if not patient_dir.is_dir() or patient_dir.name.startswith('.'):
			continue
		if not patient_dir.name.startswith("WJ-"):
			continue

		patient_raw = patient_dir.name  # 例: WJ-8
		patient_id = _normalize_wj_patient_id(patient_raw)
		patient_out_root = out_root / patient_id

		print(f"Processing patient: {patient_raw} -> {patient_id}")

		ivus_root = patient_dir / "IVUS"
		if not ivus_root.exists() or not ivus_root.is_dir():
			print(f"  [SKIP] IVUS dir not found for {patient_raw}: {ivus_root}")
			continue

		print(f"  IVUS root: {ivus_root}")

		try:
			files = [
				p for p in sorted(ivus_root.iterdir())
				if p.is_file()
				and not p.name.startswith('.')
				and p.name.upper().startswith("XA")
			]
		except Exception:
			print(f"  [WARN] Cannot list files under: {ivus_root}")
			continue

		if not files:
			print(f"  [WARN] No XA* DICOM-like files under: {ivus_root}")
			continue

		for dicom_file in files:
			# ファイル名ごとにディレクトリを分けて保存
			if dicom_file.suffix.lower() == ".dcm":
				file_key = dicom_file.stem
			else:
				file_key = dicom_file.name

			save_dir = patient_out_root / file_key
			idx = 1

			print(f"    Processing DICOM: {dicom_file} -> file {file_key}")
			try:
				ds = pydicom.dcmread(dicom_file, force=True)
			except Exception as e:
				print(f"      Error reading {dicom_file}: {e}")
				continue

			if SAVE_MODE == "meta":
				save_dicom_metadata(ds, save_dir, dicom_file)
				continue

			try:
				img = ds.pixel_array
			except Exception as e:
				print(f"      Error getting pixel_array from {dicom_file}: {e}")
				continue

			print(
				f"      shape: {img.shape}, dtype: {img.dtype}, "
				f"min: {np.min(img)}, max: {np.max(img)}"
			)

			_ = SAVE_SEQUENTIAL_FUNC(img, save_dir, start_index=idx)


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
		help=(
			"出力 txt のパス。省略時は /workspace/data/preprocessed/IVUS_csv/IVUS_error_<target>.txt"
		),
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="出力 txt を上書きする (デフォルトは追記)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	# 出力先
	if args.output is None:
		# デフォルトは同一ファイルに追記していく
		out_path = BASE_SAVE_DIR_ERROR / f"IVUS_error_{args.target}.txt"
	else:
		out_path = Path(args.output)

	print(f"[ERROR-SCAN] target={args.target}")
	print(f"[ERROR-SCAN] output={out_path} (mode={'overwrite' if args.overwrite else 'append'})")

	with ErrorWriter(out_path, overwrite=args.overwrite) as ew:
		if args.target == "ALL":
			for t in _TARGETS:
				print(f"[ERROR-SCAN] scanning {t} ...")
				scan_target(t, ew)
		else:
			scan_target(args.target, ew)

	print("[ERROR-SCAN] done")


if __name__ == "__main__":
	main()

