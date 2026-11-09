from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.data.upload_validation import read_labeled_zip, validate_image_bytes
from src.inference.model_loader import load_keras_model
from src.inference.predict import (
    make_feature_extractor,
    predict_incremental,
    predict_keras,
)
from src.models.classifier import IncrementalClassifier
from src.training.common import load_config
from src.training.continual_learning import ContinualLearningManager


CONFIG_PATH = Path(
    os.environ.get("WASTE_CONFIG_PATH", "configs/config_mwcd.yaml")
).expanduser()
st.set_page_config(
    page_title="Phân loại rác",
    page_icon=":material/recycling:",
    layout="wide",
)


@st.cache_resource
def load_saved_model(path: str) -> tf.keras.Model:
    return load_keras_model(path)


@st.cache_resource
def load_incremental_predictor(
    base_model_path: str, classifier_path: str
) -> tuple[tf.keras.Model, IncrementalClassifier]:
    model = load_keras_model(base_model_path)
    return (
        make_feature_extractor(model),
        IncrementalClassifier.load(classifier_path),
    )


def resolve(path: str) -> Path:
    return Path(path).expanduser()


def class_display_name(config: dict, class_name: str) -> str:
    return config["data"].get("class_display_names", {}).get(class_name, class_name)


def configured_manager(config: dict) -> ContinualLearningManager:
    data = config["data"]
    continual = config["continual"]
    return ContinualLearningManager(
        continual["output_dir"],
        tuple(data["class_names"]),
        int(data["image_size"]),
        seed=int(config["project"]["seed"]),
        replay_images_per_class=int(continual["replay_images_per_class"]),
        replay_images_per_update=int(continual["replay_images_per_update"]),
        max_old_validation_accuracy_drop=float(
            continual["max_old_validation_accuracy_drop"]
        ),
        minimum_new_validation_accuracy=float(
            continual["minimum_new_validation_accuracy"]
        ),
        test_fraction=float(data["test_fraction"]),
        validation_fraction=float(data["validation_fraction"]),
    )


def model_artifacts(config: dict) -> list[Path]:
    paths = set()
    for item in config["models"].values():
        directory = resolve(item["output_dir"])
        if directory.is_dir():
            for path in directory.glob("*.keras"):
                if ".checkpoint." in path.name:
                    continue
                metadata_path = path.with_suffix(".json")
                if metadata_path.is_file():
                    with metadata_path.open(encoding="utf-8") as file:
                        metadata = json.load(file)
                    if metadata.get("candidate_is_recommended") is False:
                        continue
                paths.add(path)
    return sorted(paths, key=lambda path: path.stat().st_mtime, reverse=True)


def read_pending(staging_dir: Path) -> list[dict]:
    if not staging_dir.is_dir():
        return []
    entries = []
    for metadata_path in sorted(staging_dir.glob("*/metadata.json")):
        with metadata_path.open(encoding="utf-8") as file:
            entry = json.load(file)
        filename = entry.get("filename")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}
        ):
            raise ValueError(f"Invalid staged image filename in {metadata_path}")
        entry["path"] = metadata_path.parent / filename
        if entry["path"].resolve().parent != metadata_path.parent.resolve():
            raise ValueError(f"Staged image path escapes its record folder: {metadata_path}")
        if entry["path"].is_file() and entry.get("confirmed") is True:
            entries.append(entry)
    return entries


def overview(config: dict, manager: ContinualLearningManager) -> None:
    st.title("Phân loại rác bằng hình ảnh", icon=":material/recycling:")
    st.caption(
        "Nhận diện nhanh, quản lý mô hình rõ ràng và hỗ trợ bổ sung dữ liệu "
        "được xác nhận."
    )
    artifacts = model_artifacts(config)
    with st.container(border=True):
        st.subheader("Không gian phân loại")
        st.write(
            f"Dataset: **{config['project']['name']}** · "
            f"**{len(config['data']['class_names'])} nhóm rác** · "
            f"**{len(artifacts)} model sẵn sàng**"
        )
        if artifacts:
            metadata_path = artifacts[0].with_suffix(".json")
            with metadata_path.open(encoding="utf-8") as file:
                model_metadata = json.load(file)
            metrics = model_metadata.get("test_metrics", {})
            if model_metadata.get("class_names") == config["data"]["class_names"]:
                split_sizes = model_metadata.get("split_sizes", {})
                train_count = split_sizes.get("train")
                validation_count = split_sizes.get("validation")
                test_count = split_sizes.get("test")
                if all(
                    isinstance(value, int)
                    for value in (train_count, validation_count, test_count)
                ):
                    total_count = train_count + validation_count + test_count
                    inventory = model_metadata.get("dataset_inventory", {})
                    source_count = inventory.get("source_image_files")
                    duplicate_count = inventory.get(
                        "exact_duplicate_images_excluded"
                    )
                    if (
                        isinstance(source_count, int)
                        and isinstance(duplicate_count, int)
                    ):
                        source_column, duplicate_column, usable_column = st.columns(3)
                        source_column.metric("Ảnh trong dataset", f"{source_count:,}")
                        duplicate_column.metric(
                            "Ảnh trùng đã loại", f"{duplicate_count:,}"
                        )
                        usable_column.metric(
                            "Ảnh duy nhất dùng", f"{total_count:,}"
                        )
                    else:
                        st.metric("Tổng ảnh đã chia tập", f"{total_count:,}")
                    train_column, validation_column, test_column = st.columns(3)
                    train_column.metric("Train", f"{train_count:,}")
                    validation_column.metric("Validation", f"{validation_count:,}")
                    test_column.metric("Test", f"{test_count:,}")
                accuracy = metrics.get("accuracy")
                macro_f1 = metrics.get("f1_macro")
                if all(
                    isinstance(value, (int, float)) for value in (accuracy, macro_f1)
                ):
                    accuracy_column, f1_column = st.columns(2)
                    accuracy_column.metric(
                        "Accuracy test (Colab)", f"{accuracy:.2%}"
                    )
                    f1_column.metric("Macro F1 test", f"{macro_f1:.2%}")
                    st.caption(
                        "Accuracy và Macro F1 theo JSON của model, được tính "
                        f"trên {test_count:,} ảnh test; không phải độ chính xác "
                        "được đảm bảo cho từng ảnh."
                    )
        st.caption(
            "Điểm softmax chỉ là điểm số của mô hình, chưa được hiệu chuẩn "
            "thành xác suất tin cậy."
        )
    if manager.initialized:
        try:
            metadata = manager.active_metadata()
        except ValueError as error:
            st.warning(
                "Cấu hình lớp/kích thước ảnh hiện tại khác với classifier đang "
                "hoạt động. Mở mục Học tăng dần để xem hướng dẫn tái khởi tạo."
            )
            st.caption(str(error))
        else:
            with st.container(border=True):
                st.subheader("Mô hình học tăng dần đang hoạt động")
                st.caption(f"Phiên bản {metadata['version']}")
                left, middle, right = st.columns(3)
                left.metric("Số lớp", len(metadata["class_names"]))
                middle.metric(
                    "Độ chính xác test lưu gần nhất",
                    f"{metadata['test_metrics']['accuracy']:.1%}",
                )
                right.metric("Cập nhật", metadata["created_at"][:10])
    else:
        st.caption("Có thể nhận diện ngay; học tăng dần là tính năng tùy chọn.")
    st.subheader("Các nhóm rác")
    st.markdown(
        " · ".join(
            f"`{class_display_name(config, name)}`"
            for name in config["data"]["class_names"]
        )
    )


def prediction_page(config: dict, manager: ContinualLearningManager) -> None:
    st.title("Nhận diện rác", icon=":material/center_focus_strong:")
    artifacts = model_artifacts(config)
    use_incremental = manager.initialized
    options = ["Classifier continual đang hoạt động"] if use_incremental else []
    options.extend(str(path) for path in artifacts)
    if not options:
        st.info("Chưa có mô hình. Huấn luyện CNN hoặc MobileNetV2 trước.")
        return
    def model_label(option: str) -> str:
        if option == "Classifier continual đang hoạt động":
            return option
        model_path = Path(option)
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as file:
                metadata = json.load(file)
            if metadata.get("model") == "mobilenet_v2_balanced_fine_tune":
                status = (
                    "ứng viên đạt tiêu chí validation"
                    if metadata.get("quality_gate", {}).get("passed")
                    else "ứng viên chưa đạt tiêu chí validation"
                )
                return f"{model_path.name} ({status})"
        return model_path.name

    selected = st.selectbox("Mô hình", options, format_func=model_label)
    uploaded = st.file_uploader(
        "Chọn ảnh rác", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded is None:
        return
    max_bytes = int(config["app"]["max_upload_mb"]) * 1024 * 1024
    contents = uploaded.getvalue()
    if len(contents) > max_bytes:
        st.error(f"Ảnh vượt giới hạn {config['app']['max_upload_mb']} MB.")
        return
    try:
        with Image.open(io.BytesIO(contents)) as candidate:
            candidate.verify()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        st.error("File tải lên không phải ảnh hợp lệ hoặc ảnh đã bị hỏng.")
        return
    st.image(image, caption="Ảnh tải lên", width=360, alt="Ảnh rác được tải lên")
    if st.button("Phân loại ảnh"):
        with st.spinner("Đang phân loại..."):
            try:
                if selected == "Classifier continual đang hoạt động":
                    metadata = manager.active_metadata()
                    predictor, classifier = load_incremental_predictor(
                        metadata["base_model_path"],
                        str(
                            (manager.active_version_dir / "classifier.joblib").resolve()
                        ),
                    )
                    names = list(config["data"]["class_names"])
                else:
                    model_path = Path(selected)
                    model = load_saved_model(str(model_path.resolve()))
                    metadata_path = model_path.with_suffix(".json")
                    if not metadata_path.is_file():
                        raise FileNotFoundError(
                            f"Metadata model is missing: {metadata_path}"
                        )
                    with metadata_path.open(encoding="utf-8") as file:
                        names = json.load(file)["class_names"]
                started = time.perf_counter()
                if selected == "Classifier continual đang hoạt động":
                    scores = predict_incremental(
                        predictor,
                        classifier,
                        image,
                        int(config["data"]["image_size"]),
                    )
                else:
                    scores = predict_keras(
                        model, image, int(config["data"]["image_size"])
                    )
                elapsed = (time.perf_counter() - started) * 1000
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                st.error(f"Không thể dự đoán: {error}")
                return
        names = [class_display_name(config, name) for name in names]
        best = int(scores.argmax())
        st.subheader(f"Dự đoán: {names[best]}")
        st.metric("Điểm cao nhất (chưa hiệu chuẩn)", f"{scores[best]:.1%}")
        if scores[best] < 0.5:
            st.warning(
                "Điểm dự đoán thấp; hãy xem lại ảnh và không dùng nhãn dự đoán "
                "làm nhãn đúng nếu chưa xác nhận."
            )
        st.caption(f"Thời gian suy luận: {elapsed:.1f} ms")
        st.bar_chart(pd.Series(scores, index=names, name="Điểm mô hình"))


def additional_data_page(config: dict) -> None:
    st.title("Bổ sung dữ liệu", icon=":material/add_photo_alternate:")
    st.caption(
        "Tải ảnh riêng lẻ cùng nhãn bạn xác nhận, hoặc ZIP có cấu trúc thư mục "
        "theo lớp. Dự đoán mô hình không tự động trở thành nhãn huấn luyện."
    )
    class_names = list(config["data"]["class_names"])
    staging_root = resolve(config["app"]["staging_dir"])
    max_upload_bytes = int(config["app"]["max_upload_mb"]) * 1024 * 1024
    individual_tab, archive_tab = st.tabs(["Ảnh riêng lẻ", "File ZIP theo lớp"])

    with individual_tab:
        selected_class = st.selectbox(
            "Nhãn đúng đã xác nhận",
            class_names,
            format_func=lambda name: class_display_name(config, name),
        )
        uploads = st.file_uploader(
            "Chọn một hoặc nhiều ảnh cùng nhãn",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
            key="individual_images",
        )
        if uploads:
            if len(uploads) > int(config["app"]["max_zip_files"]):
                st.error(
                    f"Chỉ nhận tối đa {config['app']['max_zip_files']} ảnh mỗi lần."
                )
                return
            st.caption(
                f"{len(uploads)} ảnh. Giới hạn mỗi ảnh: "
                f"{config['app']['max_upload_mb']} MB."
            )
            preview = st.columns(min(4, len(uploads)))
            valid_uploads = []
            for index, upload in enumerate(uploads):
                content = upload.getvalue()
                try:
                    validate_image_bytes(content, max_upload_bytes)
                    with Image.open(io.BytesIO(content)) as image:
                        preview[index % len(preview)].image(
                            image.convert("RGB"),
                            caption=upload.name,
                            width=160,
                        )
                    valid_uploads.append((upload.name, selected_class, content))
                except ValueError as error:
                    st.error(f"{upload.name}: {error}")
            if st.button("Lưu ảnh riêng lẻ đã xác nhận", type="primary"):
                saved = save_confirmed_uploads(staging_root, valid_uploads)
                st.success(f"Đã lưu {saved} ảnh có nhãn do người dùng xác nhận.")

    with archive_tab:
        st.caption(
            "Dùng thư mục `class-name/image.jpg`, có thể nằm trong một thư mục "
            "bao ngoài. ZIP được đọc trực tiếp, không giải nén đường dẫn ra đĩa."
        )
        archive = st.file_uploader("Chọn ZIP dataset", type=["zip"], key="labeled_zip")
        if archive:
            archive_bytes = archive.getvalue()
            max_zip_bytes = int(config["app"]["max_zip_upload_mb"]) * 1024 * 1024
            if len(archive_bytes) > max_zip_bytes:
                st.error(
                    f"ZIP vượt giới hạn tải lên "
                    f"{config['app']['max_zip_upload_mb']} MB."
                )
                return
            try:
                labeled_images = read_labeled_zip(
                    archive_bytes,
                    class_names,
                    max_upload_bytes,
                    max_files=int(config["app"]["max_zip_files"]),
                    max_total_uncompressed_bytes=(
                        max_upload_bytes
                        * int(config["app"]["max_zip_expansion_factor"])
                    ),
                )
            except ValueError as error:
                st.error(f"ZIP không hợp lệ: {error}")
                return
            st.write(f"Ảnh hợp lệ, duy nhất: {len(labeled_images)}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Đường dẫn trong ZIP": name, "Nhãn theo thư mục": label}
                        for name, label, _ in labeled_images
                    ]
                ),
                width="stretch",
            )
            first_images = st.columns(min(4, len(labeled_images)))
            for index, (name, label, content) in enumerate(labeled_images[:12]):
                with Image.open(io.BytesIO(content)) as image:
                    first_images[index % len(first_images)].image(
                        image.convert("RGB"),
                        caption=f"{label}: {Path(name).name}",
                        width=160,
                    )
            confirm_labels = st.checkbox(
                "Tôi xác nhận nhãn của từng ảnh theo tên thư mục lớp trong ZIP."
            )
            if st.button(
                "Lưu ZIP vào vùng chờ",
                type="primary",
                disabled=not confirm_labels,
            ):
                saved = save_confirmed_uploads(staging_root, labeled_images)
                st.success(f"Đã lưu {saved} ảnh có nhãn đã xác nhận.")


def save_confirmed_uploads(
    staging_root: Path, uploads: list[tuple[str, str, bytes]]
) -> int:
    saved = 0
    for source_name, class_name, contents in uploads:
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        item_dir = staging_root / uuid4().hex
        item_dir.mkdir(parents=True, exist_ok=False)
        filename = f"image{suffix}"
        (item_dir / filename).write_bytes(contents)
        metadata = {
            "filename": filename,
            "class_name": class_name,
            "source_name": Path(source_name).name,
            "confirmed": True,
        }
        with (item_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        saved += 1
    return saved


def continual_page(config: dict, manager: ContinualLearningManager) -> None:
    st.title("Học tăng dần và phiên bản", icon=":material/auto_awesome:")
    if not manager.initialized:
        st.warning(
            "Chưa có classifier. Huấn luyện MobileNetV2 và đặt dataset ban đầu "
            "đúng đường dẫn cấu hình trước khi khởi tạo."
        )
        candidates = model_artifacts(config)
        mobilenet_candidates = [
            path
            for path in candidates
            if path.name.startswith("mobilenet_")
        ]
        if not mobilenet_candidates:
            st.info("Chưa tìm thấy MobileNetV2 đã huấn luyện trong thư mục models.")
            return
        selected_model = st.selectbox(
            "Mô hình MobileNetV2 làm bộ trích xuất đặc trưng",
            mobilenet_candidates,
            format_func=lambda path: path.name,
        )
        if st.button("Khởi tạo classifier từ dataset"):
            try:
                with st.spinner("Đang tạo feature và cố định tập test..."):
                    metadata = manager.initialize(
                        selected_model, config["data"]["dataset_dir"]
                    )
                st.success(
                    f"Khởi tạo thành công. Test accuracy: "
                    f"{metadata['test_metrics']['accuracy']:.1%}"
                )
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Khởi tạo thất bại: {error}")
        return

    pending = read_pending(resolve(config["app"]["staging_dir"]))
    try:
        metadata = manager.active_metadata()
    except ValueError as error:
        st.error(f"Cấu hình không khớp phiên bản đang hoạt động: {error}")
        st.warning(
            "Nếu bạn vừa thêm lớp rác, hãy cập nhật data.class_names và cung cấp "
            "dataset có nhãn cho tất cả lớp cũ lẫn lớp mới. Có thể khởi tạo "
            "classifier mới trên feature MobileNetV2 hiện có; phiên bản cũ vẫn "
            "được lưu để khôi phục."
        )
        candidates = [
            path
            for path in model_artifacts(config)
            if path.name.startswith("mobilenet_")
        ]
        if not candidates:
            st.info("Cần có model MobileNetV2 để trích xuất đặc trưng.")
            return
        base_model = st.selectbox(
            "Model MobileNetV2",
            candidates,
            format_func=lambda path: path.name,
            key="reinitialize_model",
        )
        confirmed = st.checkbox(
            "Tôi xác nhận dataset cấu hình có nhãn đúng cho mọi lớp cũ và mới."
        )
        if st.button(
            "Tạo phiên bản classifier mới",
            type="primary",
            disabled=not confirmed,
        ):
            try:
                with st.spinner("Đang khởi tạo và kiểm tra classifier mới..."):
                    result = manager.initialize(
                        base_model,
                        config["data"]["dataset_dir"],
                        force=True,
                        class_sources=config["data"].get("class_sources"),
                    )
                st.success(
                    f"Đã tạo phiên bản {result['version']}; phiên bản cũ vẫn được "
                    "giữ trong lịch sử."
                )
            except (ValueError, RuntimeError, OSError) as update_error:
                st.error(f"Không thể tạo classifier mới: {update_error}")
        return
    st.metric("Ảnh đã xác nhận đang chờ", len(pending))
    st.metric("Ảnh trong replay buffer", sum(metadata.get("replay_counts", {}).values()))
    if pending:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Tên file": item["source_name"],
                        "Nhãn": class_display_name(config, item["class_name"]),
                    }
                    for item in pending
                ]
            ),
            width="stretch",
        )
        if st.button("Cập nhật classifier", type="primary"):
            names = tuple(config["data"]["class_names"])
            label_map = {name: index for index, name in enumerate(names)}
            records = []
            for item in pending:
                if item["class_name"] not in label_map:
                    st.error(f"Nhãn chưa cấu hình: {item['class_name']}")
                    return
                records.append((item["path"], label_map[item["class_name"]]))
            try:
                with st.spinner("Đang đánh giá và tạo phiên bản tạm..."):
                    result = manager.update(records)
                if result["accepted"]:
                    st.success(
                        f"Đã kích hoạt {result['version']}. Test accuracy: "
                        f"{result['before_test_metrics']['accuracy']:.1%} → "
                        f"{result['after_test_metrics']['accuracy']:.1%}."
                    )
                else:
                    st.warning(
                        f"Phiên bản {result['version']} bị từ chối; phiên bản đang "
                        "hoạt động không thay đổi. Xem metric trong lịch sử."
                    )
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Cập nhật thất bại; phiên bản hiện hoạt động được giữ nguyên: {error}")
    else:
        st.info("Hãy thêm ảnh và xác nhận nhãn tại mục Bổ sung dữ liệu trước.")

    st.subheader("Lịch sử phiên bản")
    versions = manager.versions()
    if versions:
        rows = [
            {
                "Phiên bản": item["version"],
                "Trạng thái": (
                    "Đang hoạt động"
                    if item["is_active"]
                    else "Đã chấp nhận"
                    if item["status"] == "active"
                    else item["status"]
                ),
                "Test accuracy": item.get("after_test_metrics", item.get("test_metrics", {})).get("accuracy"),
                "Validation cũ": item.get("after_fixed_validation_metrics", item.get("validation_metrics", {})).get("accuracy"),
                "Validation mới": item.get("after_new_validation_metrics", {}).get("accuracy"),
                "Thời gian cập nhật (s)": item.get("update_seconds"),
                "Tạo lúc": item["created_at"],
            }
            for item in versions
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        old_versions = [
            item for item in versions if not item["is_active"] and item["status"] != "rejected"
        ]
        if old_versions:
            choice = st.selectbox(
                "Khôi phục phiên bản đã chấp nhận",
                [item["version"] for item in old_versions],
            )
            if st.button("Khôi phục phiên bản"):
                try:
                    manager.rollback(choice)
                    st.success(f"Đã khôi phục phiên bản {choice}.")
                except (ValueError, OSError) as error:
                    st.error(f"Không thể khôi phục: {error}")


def reports_page(config: dict, manager: ContinualLearningManager) -> None:
    st.title("Báo cáo", icon=":material/analytics:")
    result_dir = resolve(config["results"]["output_dir"])
    metric_files = sorted(result_dir.glob("*_metrics.json")) if result_dir.is_dir() else []
    metrics = []
    for path in metric_files:
        with path.open(encoding="utf-8") as file:
            metrics.append(json.load(file))
    known_models = {item.get("model") for item in metrics}
    for model_path in model_artifacts(config):
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            continue
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        model_metrics = metadata.get("test_metrics")
        if (
            metadata.get("class_names") == config["data"]["class_names"]
            and isinstance(model_metrics, dict)
            and model_metrics.get("model") not in known_models
        ):
            metrics.append(model_metrics)
            known_models.add(model_metrics.get("model"))
    if metrics:
        frame = pd.DataFrame(
            [
                {
                    "Mô hình": item["model"],
                    "Accuracy": item["accuracy"],
                    "Precision macro": item["precision_macro"],
                    "Recall macro": item["recall_macro"],
                    "F1 macro": item["f1_macro"],
                    "Thời gian huấn luyện (s)": item.get("training_seconds"),
                    "Suy luận mỗi ảnh (ms)": item.get("inference_ms_per_image"),
                    "Tập đánh giá": item["evaluation_split"],
                }
                for item in metrics
            ]
        )
        st.dataframe(frame, width="stretch")
        st.download_button(
            "Tải metric CSV",
            frame.to_csv(index=False).encode("utf-8-sig"),
            file_name="model_metrics.csv",
            mime="text/csv",
        )
        st.download_button(
            "Tải metric JSON",
            json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="model_metrics.json",
            mime="application/json",
        )
        selected = st.selectbox("Ma trận nhầm lẫn", [item["model"] for item in metrics])
        chosen = next(item for item in metrics if item["model"] == selected)
        st.dataframe(
            pd.DataFrame(
                chosen["confusion_matrix"],
                index=[
                    class_display_name(config, name)
                    for name in config["data"]["class_names"]
                ],
                columns=[
                    class_display_name(config, name)
                    for name in config["data"]["class_names"]
                ],
            ),
            width="stretch",
        )
    else:
        st.info("Chưa có báo cáo đánh giá; hãy huấn luyện mô hình trước.")

    if manager.initialized:
        versions = manager.versions()
        if versions:
            version_frame = pd.DataFrame(
                [
                    {
                        "version": item["version"],
                        "active": item["is_active"],
                        "accepted": item.get("accepted", True),
                        "test_accuracy": item.get("after_test_metrics", item.get("test_metrics", {})).get("accuracy"),
                    }
                    for item in versions
                ]
            )
            st.subheader("Lịch sử học tăng dần")
            st.dataframe(version_frame, width="stretch")
            st.download_button(
                "Tải lịch sử phiên bản JSON",
                json.dumps(versions, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="continual_versions.json",
                mime="application/json",
            )


def main() -> None:
    if not CONFIG_PATH.is_file():
        st.error(f"Không tìm thấy cấu hình `{CONFIG_PATH}`.")
        return
    config = load_config(CONFIG_PATH)
    manager = configured_manager(config)
    pages = {
        "Tổng quan": lambda: overview(config, manager),
        "Nhận diện rác": lambda: prediction_page(config, manager),
        "Bổ sung dữ liệu": lambda: additional_data_page(config),
        "Học tăng dần": lambda: continual_page(config, manager),
        "Báo cáo": lambda: reports_page(config, manager),
    }
    with st.sidebar:
        st.title("Tái chế thông minh")
        st.caption(f"Cấu hình: {CONFIG_PATH.name}")
        st.radio(
            "Điều hướng",
            list(pages),
            format_func=lambda label: {
                "Tổng quan": ":material/dashboard: Tổng quan",
                "Nhận diện rác": ":material/center_focus_strong: Nhận diện rác",
                "Bổ sung dữ liệu": ":material/add_photo_alternate: Bổ sung dữ liệu",
                "Học tăng dần": ":material/auto_awesome: Học tăng dần",
                "Báo cáo": ":material/analytics: Báo cáo",
            }[label],
            key="main_navigation",
        )
        st.caption("Phân loại có hỗ trợ AI")
        selected = st.session_state.main_navigation
    pages[selected]()


if __name__ == "__main__":
    main()
