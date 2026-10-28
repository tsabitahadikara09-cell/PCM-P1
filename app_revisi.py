    s = segmentasi_lesi(kelas)
    label_img = measure.label(s["lesi"])
    props = measure.regionprops(label_img)
    return s, label_img, (props[0] if props else None)


def titik_feret(mask):
    kontur = np.vstack(measure.find_contours(mask.astype(float), 0.5))
    hull = kontur[ConvexHull(kontur).vertices]
    jarak = squareform(pdist(hull))
    i, j = np.unravel_index(np.argmax(jarak), jarak.shape)
    return hull[i], hull[j]


@st.cache_data(show_spinner="Menguji filter terhadap noise...")
def uji_noise(kelas, sp=0.05, sg=20, sv=0.04):
    gray = ke_gray(SAMPEL[kelas])
    patch, lesi = cari_patch_seragam(gray), daerah_lesi_kasar(gray)
    noisy = {"Salt & pepper": noise_salt_pepper(gray, sp), "Gaussian": noise_gaussian(gray, sg),
             "Speckle": noise_speckle(gray, sv)}
    hasil, baris = {}, []
    for nama_noise, citra_noise in noisy.items():
        hasil[nama_noise] = filter_3x3(citra_noise)
        baris.append({"Noise": nama_noise, "Filter": "Tanpa filter", **evaluasi(gray, citra_noise, patch, lesi)})
        for nama_filter, g in hasil[nama_noise].items():
            baris.append({"Noise": nama_noise, "Filter": nama_filter, **evaluasi(gray, g, patch, lesi)})
    return noisy, hasil, pd.DataFrame(baris)


@st.cache_data(show_spinner=False)
def clahe_semua(clip):
    return {k: clahe_rgb(i, clip) for k, i in SAMPEL.items()}


@st.cache_data(show_spinner="Menghitung mean filter from scratch...")
def mean_scratch(kelas):
    g = ke_gray(SAMPEL[kelas]).astype(int)
    return g, MeanFilter(g, 9), ndi.convolve(g.astype(float), np.full((3, 3), 1 / 9))


# ============================================================================
# Halaman-halaman (sesuai bagian pada notebook)
# ============================================================================
def sec_data():
    header("1 · Data & Contoh Citra", "Pratinjau citra yang diunggah atau sampel dataset yang dipilih.")
    contoh = next(iter(SAMPEL.values()))
    m = st.columns(4)
    m[0].metric("Jumlah citra", len(SAMPEL))
    m[1].metric("Ukuran citra pertama", f"{contoh.shape[0]} × {contoh.shape[1]}")
    m[2].metric("Channel warna", contoh.shape[2] if contoh.ndim == 3 else 1)
    m[3].metric("Tipe data", str(contoh.dtype))
    ncols = min(4, len(SAMPEL))
    nrows = int(np.ceil(len(SAMPEL) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows), squeeze=False)
    for ax, (k, img) in zip(axes.ravel(), SAMPEL.items()):
        ax.imshow(img)
        ax.set_title(f"Citra {k}\n{NAMA[str(k)]}", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(SAMPEL):]:
        ax.axis("off")
    fig.suptitle("Pratinjau Citra untuk Analisis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.8)
    show(fig)
    st.dataframe(pd.DataFrame([{"ID citra": k, "Nama file / kelas": NAMA[str(k)], "dtype": str(i.dtype), "shape": str(i.shape),
                                "min": int(i.min()), "max": int(i.max())} for k, i in SAMPEL.items()]).set_index("ID citra"))


def sec_hist():
    header("2 · Histogram RGB & Equalization",
           "Tiap channel dipisah → citra grayscale + histogram, lalu diequalisasi memakai cumulative histogram (CDF).")
    st.latex(r"I_{eq}(u,v)=\frac{CDF\left(I(u,v)\right)}{CDF_{max}}\times 255")
    img, nama = SAMPEL[kelas], f"Kelas {kelas}: {NAMA[str(kelas)]}"
    eq = equalize_rgb(img)
    t1, t2, t3 = st.tabs(["🖼️ Citra asli", "✨ Equalized RGB", "📋 Brightness & contrast"])
    with t1:
        show(plot_split_channel(img, f"Citra Asli\n{nama}", "Histogram"))
    with t2:
        show(plot_split_channel(eq, f"Equalized RGB\n{nama}", "Histogram Equalized"))
    with t3:
        st.caption("Brightness = rata-rata intensitas · Contrast = standar deviasi intensitas.")
        st.dataframe(pd.DataFrame({
            "Channel": [n for n, _ in CHANNEL],
            "Brightness asli": [img[:, :, i].mean() for i in range(3)],
            "Brightness equalized": [eq[:, :, i].mean() for i in range(3)],
            "Contrast asli (std)": [img[:, :, i].std() for i in range(3)],
            "Contrast equalized (std)": [eq[:, :, i].std() for i in range(3)],
            "Rentang asli": [f"{img[:, :, i].min()} - {img[:, :, i].max()}" for i in range(3)],
            "Rentang equalized": [f"{eq[:, :, i].min()} - {eq[:, :, i].max()}" for i in range(3)],
        }).round(2).set_index("Channel"))


def sec_clahe():
    header("2b · CLAHE (Adaptive Histogram Equalization)",
           "Histogram dihitung per area kecil sehingga kontras lokal meningkat, berbeda dari equalization global.")
    clip = st.slider("Clip limit", 0.001, 0.050, 0.010, 0.001, format="%.3f")
    hasil = clahe_semua(clip)
    fig, axes = plt.subplots(2, len(hasil), figsize=(3.2 * len(hasil), 7))
    for j, (k, c) in enumerate(hasil.items()):
        axes[0, j].imshow(SAMPEL[k])
        axes[0, j].set_title(f"Asli - Kelas {k}\n{W(NAMA[str(k)], 16)}", fontsize=9)
        axes[1, j].imshow(c)
        axes[1, j].set_title(f"CLAHE - Kelas {k}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    show(fig)


def sec_teori():
    header("3 · Filtering: Teori & Contoh Perhitungan",
           "Filter menggabungkan nilai sebuah piksel dengan piksel-piksel tetangganya.")
    st.markdown("""
| Jenis | Filter | Cara kerja | Fungsi Python |
|---|---|---|---|
| Linear | **Mean** | rata-rata piksel + tetangga, bobot 1/9 (3×3) | `ndi.convolve` |
| Linear | **Gaussian** | rata-rata berbobot, bobot terbesar di pusat kernel | `ndi.gaussian_filter` |
| Non-linear | **Median** | diganti median tetangga; baik untuk *salt & pepper*, tepi tetap tajam | `ndi.median_filter` |
| Non-linear | **Min** | nilai terkecil tetangga, menghilangkan *salt noise* | `ndi.minimum_filter` |
| Non-linear | **Max** | nilai terbesar tetangga, menghilangkan *pepper noise* | `ndi.maximum_filter` |
""")
    st.markdown("##### 📐 Parameter evaluasi (*f* = citra asli, *g* = hasil filter)")
    rumus = [("MSE", r"\frac{1}{MN}\sum\sum (f-g)^2", "makin kecil, makin mirip asli"),
             ("PSNR", r"10\log_{10}\frac{255^2}{MSE}", "makin besar, makin baik"),
             ("ENL", r"\frac{M^2}{SD^2}", "makin besar, noise makin lemah"),
             ("CNR", r"\frac{|\mu_1-\mu_2|}{\sqrt{\sigma_1^2+\sigma_2^2}}", "lesi vs kulit; makin besar makin mudah dipisah"),
             ("NM", r"\frac{M_{filtered}}{M_{original}}", "mendekati 1 = informasi terjaga")]
    for col, (n, f, ket) in zip(st.columns(5), rumus):
        with col.container(border=True):
            st.markdown(f"**{n}**")
            st.latex(f)
            st.caption(ket)

    st.markdown("---")
    st.subheader("3.1 Contoh perhitungan Mean & Median filter")
    blok = np.array([[104, 100, 108], [99, 106, 98], [95, 90, 85]])
    contoh = np.array([[5, 6, 5, 7, 8], [9, 10, 1, 3, 2], [5, 6, 10, 3, 2], [7, 4, 4, 5, 1], [2, 3, 5, 3, 3]])
    med = contoh.copy()
    for j in range(1, contoh.shape[0] - 1):
        for i in range(1, contoh.shape[1] - 1):
            med[j, i] = np.median(contoh[j - 1:j + 2, i - 1:i + 2])
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Mean filter 3×3** (kernel 1/9)")
        st.dataframe(pd.DataFrame(blok))
        st.metric("Hasil pada piksel pusat (106)", f"{(blok * np.full((3, 3), 1 / 9)).sum():.4f}")
    with b.container(border=True):
        st.markdown("**Median filter 3×3**")
        c1, c2 = st.columns(2)
        c1.caption("Input")
        c1.dataframe(pd.DataFrame(contoh))
        c2.caption("Hasil")
        c2.dataframe(pd.DataFrame(med))
        st.caption(f"F(2,2) = median {sorted(contoh[0:3, 0:3].ravel().tolist())} = {med[1, 1]}")

    st.markdown("##### Mean filter *from scratch* vs `ndi.convolve`")
    gray, scratch, nd = mean_scratch(kelas)
    selisih = np.abs(scratch[1:-1, 1:-1] - np.floor(nd[1:-1, 1:-1] + 1e-9))
    st.info(f"Kelas {kelas}: selisih maksimum antara kedua cara = **{selisih.max():.0f}**")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, tt in zip(axes, (gray, scratch, nd),
                          ("Grayscale asli", "Mean filter 3x3 (from scratch, tepi = 0)", "Mean filter 3x3 (ndi.convolve)")):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(tt)
        ax.axis("off")
    fig.tight_layout()
    show(fig)


def sec_noise():
    header("3.2 · Uji Filter terhadap Noise",
           "Citra grayscale diberi tiga jenis noise lalu difilter (Mean, Gaussian, Median, Min, Max; kernel 3×3).")
    with st.expander("⚙️ Parameter noise (nilai awal sama dengan notebook)"):
        c = st.columns(3)
        sp = c[0].slider("Salt & pepper (proporsi)", 0.01, 0.20, 0.03, 0.01)
        sg = c[1].slider("Gaussian (σ)", 5, 60, 10)
        sv = c[2].slider("Speckle (varians)", 0.01, 0.20, 0.02, 0.01)
    gray = ke_gray(SAMPEL[kelas])
    noisy, hasil, tabel = uji_noise(kelas, sp, sg, sv)

    st.markdown("##### 🏆 Filter dengan PSNR tertinggi")
    for col, (nn, t) in zip(st.columns(3), tabel[tabel["Filter"] != "Tanpa filter"].groupby("Noise", sort=False)):
        best = t.loc[t["PSNR (dB)"].idxmax()]
        col.metric(nn, best["Filter"])
        col.caption(f"PSNR = {best['PSNR (dB)']:.2f} dB")

    t1, t2, t3 = st.tabs(["🖼️ Citra ber-noise", "🧪 Hasil filter", "📋 Tabel metrik"])
    with t1:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Grayscale asli")
        for ax, (nama, im) in zip(axes[1:], noisy.items()):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
