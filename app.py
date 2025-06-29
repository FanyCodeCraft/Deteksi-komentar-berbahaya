import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from model_utils import classifier, sklearn_score

st.set_page_config(page_title="Deteksi Komentar Jahat", layout="wide")
st.markdown("<style>body { background-color: #0f0f0f; color: white; }</style>", unsafe_allow_html=True)

st.title("💬 Deteksi Komentar Jahat AI")
st.markdown("Masukkan beberapa komentar. Sistem akan menganalisis tingkat bahayanya (Aman, Sedang, Sangat Berbahaya).")

def bersihkan_komentar(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

def klasifikasi(score_ai, score_ml):
    if score_ai < 0.6 and score_ml < 0.6:
        return "Aman", "#4caf50"
    elif score_ai < 0.75 or score_ml < 0.75:
        return "Bahaya Sedang", "#ff9800"
    else:
        return "Sangat Berbahaya", "#f44336"

emoji_map = {
    "Aman": "✅",
    "Bahaya Sedang": "⚠️",
    "Sangat Berbahaya": "🚨"
}

user_input = st.text_area("✏️ Masukkan komentar (1 baris = 1 komentar)", height=200, placeholder="Contoh:\nKomentar pertama\nKomentar kedua")

if st.button("🔍 Analisis Sekarang"):
    comments = [c.strip() for c in user_input.split("\n") if c.strip()]

    if not comments:
        st.warning("Kamu belum memasukkan komentar apapun.")
    else:
        st.subheader("📄 Hasil Analisis")
        results = []

        try:
            sklearn_scores = sklearn_score(comments)
        except Exception as e:
            st.error(f"❌ Gagal menghitung skor ML: {str(e)}")
            st.stop()

        if sklearn_scores is None or len(sklearn_scores) == 0 or len(sklearn_scores) != len(comments):
            st.error("❌ Terjadi ketidaksesuaian antara jumlah komentar dan skor ML.")
            st.stop()

        for i, comment in enumerate(comments):
            comment = bersihkan_komentar(comment)
            if len(comment) == 0 or len(comment) > 1000:
                st.warning(f"Komentar dilewati karena terlalu pendek/panjang: '{comment[:50]}...'")
                continue

            try:
                pred = classifier(comment)[0]
                label = pred['label']
                score_ai = pred['score']

                if label in ["LABEL_0", "NOT_TOXIC", "not_toxic"]:
                    score_ai = 1 - score_ai

                score_ml = sklearn_scores[i]
                if score_ml > 1.0:
                    score_ml = score_ml / 100

                kategori, warna = klasifikasi(score_ai, score_ml)
                kategori_emoji = f"{emoji_map[kategori]} {kategori}"

                st.markdown(f"""
                **Komentar ke-{i+1}:**  
                👉 `{comment}`  
                🔖 Label AI: `{label}`  
                🧠 Skor AI: `{round(score_ai * 100, 2)}%`  
                🤖 Skor ML: `{round(score_ml * 100, 2)}%`  
                📌 Kategori: **{kategori_emoji}**
                """)

                results.append({
                    "Komentar": comment,
                    "Kategori": kategori_emoji,
                    "Skor AI": round(score_ai * 100, 2),
                    "Skor ML": round(score_ml * 100, 2),
                    "Warna": warna
                })

            except Exception as e:
                st.error(f"❌ Gagal memproses komentar: '{comment[:30]}...'
Error: {str(e)}")
                continue

        if results:
            df = pd.DataFrame(results)
            st.subheader("🦾 Tabel Hasil Lengkap")
            st.dataframe(df[["Komentar", "Kategori", "Skor AI", "Skor ML"]], use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📅 Download Hasil sebagai CSV", csv, "hasil_analisis.csv", "text/csv")

            st.subheader("📊 Diagram Distribusi Kategori")
            pie_data = df["Kategori"].value_counts()
            color_map = {
                f"{emoji_map['Aman']} Aman": "#4caf50",
                f"{emoji_map['Bahaya Sedang']} Bahaya Sedang": "#ff9800",
                f"{emoji_map['Sangat Berbahaya']} Sangat Berbahaya": "#f44336"
            }
            colors = [color_map.get(k, "#999999") for k in pie_data.index]
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%", colors=colors)
            ax.set_facecolor("#0f0f0f")
            st.pyplot(fig)
        else:
            st.warning("Tidak ada komentar yang berhasil dianalisis.")