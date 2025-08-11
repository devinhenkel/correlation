import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import gradio as gr


def load_dataframe(file) -> pd.DataFrame:
    if file is None:
        raise ValueError("No file provided")
    # Gradio may pass a file path (str) or a file-like object with a .name attribute.
    if isinstance(file, str):
        path = file
    else:
        path = getattr(file, "name", None)
        if path is None:
            raise ValueError("Unsupported file type input")
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type. Please upload CSV or XLSX.")


def compute_numeric_and_binary(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    non_numeric_cols = [c for c in df.columns if c not in numeric_df.columns]
    binary_df = (numeric_df > 3).astype(int)
    return numeric_df, binary_df, non_numeric_cols


def on_upload(file):
    try:
        df = load_dataframe(file)
        numeric_df, binary_df, non_numeric_cols = compute_numeric_and_binary(df)
        info_lines = []
        if len(non_numeric_cols) > 0:
            info_lines.append(
                "Only numeric columns are used for the correlation and binary copy. "
                f"Dropped non-numeric columns: {', '.join(map(str, non_numeric_cols))}"
            )
        if numeric_df.shape[1] == 0:
            info_lines.append("No numeric columns found. Correlation cannot be computed.")
        info_md = "\n\n".join(info_lines) if info_lines else ""
        return (
            df.head(),
            binary_df.head() if binary_df.shape[1] > 0 else pd.DataFrame(),
            info_md,
            numeric_df,  # state
            binary_df,   # state
            gr.update(value=pd.DataFrame()),  # clear correlation df display
            gr.update(value=None, visible=False, label="Download correlation as CSV", file_name="correlation_spearman.csv"),
        )
    except Exception as e:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            f"Error: {e}",
            None,
            None,
            gr.update(value=pd.DataFrame()),
            gr.update(value=None, visible=False, label="Download correlation as CSV", file_name="correlation_spearman.csv"),
        )


def corr_original(numeric_df: Optional[pd.DataFrame]):
    if numeric_df is None or getattr(numeric_df, "shape", (0, 0))[1] == 0:
        return gr.update(value=pd.DataFrame()), gr.update(value=None, visible=False)
    corr = numeric_df.corr(method="spearman")
    csv_bytes = corr.to_csv(index=True).encode("utf-8")
    return corr, gr.update(value=csv_bytes, visible=True, label="Download correlation as CSV", file_name="correlation_spearman.csv")


def corr_categorical(binary_df: Optional[pd.DataFrame]):
    if binary_df is None or getattr(binary_df, "shape", (0, 0))[1] == 0:
        return gr.update(value=pd.DataFrame()), gr.update(value=None, visible=False)
    corr = binary_df.corr(method="spearman")
    csv_bytes = corr.to_csv(index=True).encode("utf-8")
    return corr, gr.update(value=csv_bytes, visible=True, label="Download correlation as CSV", file_name="correlation_spearman.csv")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Spearman Correlation App") as demo:
        gr.Markdown("""
        # Spearman Correlation App
        Upload a CSV or XLSX file, preview the data, generate a binary copy (> 3 -> 1, else 0), and compute Spearman correlations.
        """)

        numeric_state = gr.State(None)
        binary_state = gr.State(None)

        with gr.Row():
            file_input = gr.File(label="Upload a CSV or XLSX file", file_types=[".csv", ".xlsx"]) 

        info_md = gr.Markdown(visible=True)

        with gr.Row():
            orig_head = gr.Dataframe(label="Original data (head)", interactive=False)
        with gr.Row():
            bin_head = gr.Dataframe(label="Binary copy (> 3 -> 1, else 0) (head)", interactive=False)

        with gr.Row():
            btn_orig = gr.Button("Correlate original", variant="primary")
            btn_cat = gr.Button("Correlate categorical")

        corr_df = gr.Dataframe(label="Spearman correlation matrix", interactive=False)
        download_btn = gr.DownloadButton(label="Download correlation as CSV", visible=False)

        file_input.change(
            fn=on_upload,
            inputs=file_input,
            outputs=[orig_head, bin_head, info_md, numeric_state, binary_state, corr_df, download_btn],
        )

        btn_orig.click(fn=corr_original, inputs=numeric_state, outputs=[corr_df, download_btn])
        btn_cat.click(fn=corr_categorical, inputs=binary_state, outputs=[corr_df, download_btn])

    return demo


demo = build_ui()

if __name__ == "__main__":
    demo.launch()