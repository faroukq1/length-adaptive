# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# latex
- In the thesis memoir, the hybrid model "DCH-BERT4Rec" should be renamed to "TCN-BERT4Rec" — they are the same model; replace the name everywhere rather than deleting the data. Confidence: 0.85
- When drawing TikZ flow diagrams with multiple inputs converging into one node, use direct `--` (diagonal) arrows, not `-|` (orthogonal/square) paths — diagonal arrows look cleaner. Confidence: 0.70
- Use parenthetical citations (`\citep{}`) instead of textual citations (`\citet{}` or `\cite{}`) — all references should appear within parentheses. Confidence: 0.70
- Add `\markboth{Chapter Title}{}` immediately after `\chapter*{Chapter Title}` for unnumbered chapters (e.g., General Introduction, General Conclusion) to fix running headers in the memoir class. Confidence: 0.70

# notebooks
- Notebooks in this project run on Kaggle — avoid machine-specific absolute paths and use paths relative to the notebook file (e.g., `Path(__file__).parent / "..."`). Confidence: 0.70
- Models should start training from scratch — do not use checkpoint/resume functionality to continue from previous runs. Confidence: 0.65

