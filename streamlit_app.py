import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import easyocr

# --- Funkcje pomocnicze ---

# Funkcja do obliczenia współczynnika kontrastu WCAG
def calculate_contrast_ratio(color1, color2):
    def get_luminance(rgb):
        rgb = [c/255 for c in rgb]
        rgb = [c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    luminance1 = get_luminance(color1)
    luminance2 = get_luminance(color2)
    return (luminance1 + 0.05) / (luminance2 + 0.05) if luminance1 > luminance2 else (luminance2 + 0.05) / (luminance1 + 0.05)

# Funkcja do oceny zgodności z WCAG AA i AAA
def check_wcag_compliance(contrast_ratio):
    return {
        'AA_normal': contrast_ratio >= 4.5,
        'AA_large': contrast_ratio >= 3.0,
        'AAA_normal': contrast_ratio >= 7.0,
        'AAA_large': contrast_ratio >= 4.5
    }

# Inicjalizacja czytnika EasyOCR – dodajemy komunikat diagnostyczny
@st.cache_resource
def load_ocr_reader(languages=['pl', 'en']):
    st.write("Inicjalizacja modelu EasyOCR...")
    try:
        # Upewnij się, że torch jest zaimportowany dopiero tutaj, aby uniknąć problemów przy uruchomieniu
        import torch
        gpu = torch.cuda.is_available()
    except Exception as e:
        gpu = False
    try:
        reader = easyocr.Reader(languages, gpu=gpu)
        st.write("Model EasyOCR został załadowany.")
        return reader
    except Exception as e:
        st.error(f"Błąd podczas inicjalizacji EasyOCR: {e}")
        return None

# Detekcja tekstu za pomocą EasyOCR
def detect_text_easyocr(image, reader):
    try:
        st.write("Detekcja tekstu...")
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        results = reader.readtext(img_array)
        text_regions = []
        for i, (bbox, text, conf) in enumerate(results):
            x_min = min(point[0] for point in bbox)
            y_min = min(point[1] for point in bbox)
            x_max = max(point[0] for point in bbox)
            y_max = max(point[1] for point in bbox)
            text_regions.append({
                'text': text,
                'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                'confidence': conf * 100
            })
        st.write(f"Wykryto {len(text_regions)} regionów tekstowych.")
        return text_regions
    except Exception as e:
        st.error(f"Błąd podczas detekcji tekstu: {e}")
        return []

# Pobieranie kolorów z regionu z dodatkową heurystyką
def sample_colors_from_region(image, region, num_samples=10):
    x1, y1, x2, y2 = region['bbox']
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    region_img = image[y1:y2, x1:x2]
    if region_img.size == 0:
        return None
    if len(region_img.shape) == 3:
        region_rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
    else:
        region_rgb = cv2.cvtColor(region_img, cv2.COLOR_GRAY2RGB)
    pixels = region_rgb.reshape(-1, 3)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(pixels)
    centers = km.cluster_centers_.astype(int)
    labels = km.labels_
    counts = np.bincount(labels)
    sorted_indices = np.argsort(counts)[::-1]

    def get_luminance(rgb):
        rgb = [c/255 for c in rgb]
        rgb = [c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    color1 = centers[sorted_indices[0]]
    color2 = centers[sorted_indices[1]]
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    total_pixels = len(pixels)
    dominant_threshold = 0.7
    ratio_dominant = counts[sorted_indices[0]] / total_pixels
    if ratio_dominant >= dominant_threshold:
        background_color = color1
        foreground_color = color2
    else:
        if lum1 > lum2:
            background_color = color1
            foreground_color = color2
        else:
            background_color = color2
            foreground_color = color1
    contrast_ratio = calculate_contrast_ratio(foreground_color, background_color)
    compliance = check_wcag_compliance(contrast_ratio)
    return {
        'foreground': foreground_color,
        'background': background_color,
        'contrast_ratio': contrast_ratio,
        'compliance': compliance
    }

# Sprawdzenie odległości regionu od krawędzi obrazu
def check_edge_distance(image_shape, region, margin_h_px, margin_v_px):
    try:
        x1, y1, x2, y2 = region['bbox']
        height, width = image_shape[:2]
        safe_x1 = margin_h_px
        safe_y1 = margin_v_px
        safe_x2 = width - margin_h_px
        safe_y2 = height - margin_v_px
        is_inside = (x1 >= safe_x1 and y1 >= safe_y1 and x2 <= safe_x2 and y2 <= safe_y2)
        distance_left = x1
        distance_right = width - x2
        distance_top = y1
        distance_bottom = height - y2
        percent_left = (distance_left / width) * 100
        percent_right = (distance_right / width) * 100
        percent_top = (distance_top / height) * 100
        percent_bottom = (distance_bottom / height) * 100
        return {
            'is_inside_safe_area': is_inside,
            'distances': {'left': distance_left, 'right': distance_right, 'top': distance_top, 'bottom': distance_bottom},
            'percentages': {'left': percent_left, 'right': percent_right, 'top': percent_top, 'bottom': percent_bottom}
        }
    except Exception as e:
        st.error(f"Błąd przy sprawdzaniu odległości: {e}")
        return {
            'is_inside_safe_area': False,
            'distances': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
            'percentages': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        }

# Wizualizacja wyników analizy
def visualize_results(original_image, regions_with_analysis, margin_h_px, margin_v_px):
    try:
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            vis_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            vis_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        height, width = original_image.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        safe_x1 = margin_h_px
        safe_y1 = margin_v_px
        safe_x2 = width - margin_h_px
        safe_y2 = height - margin_v_px
        for y in range(height):
            for x in range(width):
                if (x < safe_x1 or x > safe_x2 or y < safe_y1 or y > safe_y2):
                    overlay[y, x] = [0, 0, 0, 128]
        overlay[safe_y1-1:safe_y1+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        overlay[safe_y2-1:safe_y2+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        overlay[safe_y1-1:safe_y2+1, safe_x1-1:safe_x1+1] = [255, 255, 255, 200]
        overlay[safe_y1-1:safe_y2+1, safe_x2-1:safe_x2+1] = [255, 255, 255, 200]
        for region_data in regions_with_analysis:
            region = region_data['region']
            color_data = region_data['colors']
            if not color_data:
                continue
            x1, y1, x2, y2 = region['bbox']
            compliance = color_data['compliance']
            if compliance['AAA_normal']:
                box_color = [0, 255, 0, 200]
            elif compliance['AA_normal']:
                box_color = [0, 0, 255, 200]
            else:
                box_color = [255, 0, 0, 200]
            overlay[y1-1:y1+1, x1-1:x2+1] = box_color
            overlay[y2-1:y2+1, x1-1:x2+1] = box_color
            overlay[y1-1:y2+1, x1-1:x1+1] = box_color
            overlay[y1-1:y2+1, x2-1:x2+1] = box_color
        overlay_pil = Image.fromarray(overlay)
        original_pil = Image.fromarray(vis_image_rgb)
        result = Image.alpha_composite(original_pil.convert("RGBA"), overlay_pil)
        result_array = np.array(result)
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_RGBA2RGB)
        cv2.putText(result_rgb, "Strefa bezpieczna dla treści", (safe_x1 + 10, safe_y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for region_data in regions_with_analysis:
            region = region_data['region']
            color_data = region_data['colors']
            if not color_data:
                continue
            x1, y1, x2, y2 = region['bbox']
            contrast_text = f"{color_data['contrast_ratio']:.2f}"
            if color_data['compliance']['AAA_normal']:
                text_color = (0, 255, 0)
            elif color_data['compliance']['AA_normal']:
                text_color = (0, 0, 255)
            else:
                text_color = (255, 0, 0)
            cv2.putText(result_rgb, contrast_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        return result_rgb
    except Exception as e:
        st.error(f"Błąd podczas wizualizacji wyników: {e}")
        return original_image

# --- Główna aplikacja ---

def main():
    st.title("Analiza Dostępności Grafiki - WCAG")
    
    st.sidebar.header("Opcje analizy")
    languages = st.sidebar.multiselect(
        "Języki OCR",
        options=["pl", "en", "de", "fr", "es", "it"],
        default=["pl", "en"]
    )
    st.sidebar.subheader("Bezpieczny obszar")
    margin_h_percent = st.sidebar.slider("Margines poziomy (%)", 1, 25, 10,
                                           help="Minimalny odstęp od lewej i prawej krawędzi jako procent szerokości obrazu")
    margin_v_percent = st.sidebar.slider("Margines pionowy (%)", 1, 25, 10,
                                           help="Minimalny odstęp od górnej i dolnej krawędzi jako procent wysokości obrazu")
    
    uploaded_file = st.file_uploader("Wybierz plik graficzny", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        st.write("Wczytywanie obrazu...")
        pil_image = Image.open(uploaded_file)
        cv_image = np.array(pil_image)
        if len(cv_image.shape) == 2:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        st.image(pil_image, caption="Wgrana grafika", use_container_width=True)
        
        height, width = cv_image.shape[:2]
        margin_h_px = int((width * margin_h_percent) / 100)
        margin_v_px = int((height * margin_v_percent) / 100)
        st.info(f"Bezpieczny obszar: margines poziomy {margin_h_percent}% ({margin_h_px}px), margines pionowy {margin_v_percent}% ({margin_v_px}px)")
        
        st.write("Ładowanie modelu OCR...")
        reader = load_ocr_reader(languages)
        if reader is None:
            st.error("Błąd przy inicjalizacji OCR.")
            return
        
        st.write("Wykrywanie tekstu...")
        regions = detect_text_easyocr(cv_image, reader)
        if not regions:
            st.warning("Nie wykryto tekstu. Upewnij się, że obraz zawiera czytelny tekst.")
            return
        
        st.write("Analiza regionów...")
        regions_with_analysis = []
        for region in regions:
            color_data = sample_colors_from_region(cv_image, region)
            edge_data = check_edge_distance(cv_image.shape, region, margin_h_px, margin_v_px)
            regions_with_analysis.append({
                'region': region,
                'colors': color_data,
                'edge': edge_data
            })
        
        st.write("Tworzenie wizualizacji wyników...")
        result_image = visualize_results(cv_image, regions_with_analysis, margin_h_px, margin_v_px)
        st.subheader("Wyniki analizy")
        st.image(result_image, caption="Analiza dostępności", use_container_width=True)
        
        st.markdown("""
        **Legenda:**
        - **Ciemniejszy obszar** - Strefa marginesu (unikaj umieszczania tam treści)
        - **Jaśniejszy obszar** - Strefa bezpieczna dla treści
        - **Zielona ramka** - Element zgodny z WCAG AAA
        - **Niebieska ramka** - Element zgodny z WCAG AA, ale nie z AAA
        - **Czerwona ramka** - Element niezgodny z WCAG AA
        """)
        
        # Podsumowanie wyników (przykład)
        st.subheader("Podsumowanie wyników")
        valid_color_results = [r for r in regions_with_analysis if r['colors'] is not None]
        aa_pass = sum(1 for r in valid_color_results if r['colors']['compliance']['AA_normal'])
        aa_percent = (aa_pass / len(valid_color_results) * 100) if valid_color_results else 0
        aaa_pass = sum(1 for r in valid_color_results if r['colors']['compliance']['AAA_normal'])
        aaa_percent = (aaa_pass / len(valid_color_results) * 100) if valid_color_results else 0
        outside_safe_area = sum(1 for r in regions_with_analysis if not r['edge']['is_inside_safe_area'])
        outside_percent = (outside_safe_area / len(regions_with_analysis) * 100) if regions_with_analysis else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div style="background-color:#262730;padding:20px;border-radius:6px;margin-bottom:10px;border:1px solid #4B5563;">
                <div style="font-size:24px;font-weight:bold;color:#FFFFFF;">{aa_pass}/{len(valid_color_results)} ({aa_percent:.1f}%)</div>
                <div style="font-size:14px;color:#E0E0E0;margin-top:5px;">WCAG AA (normalny tekst)</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background-color:#262730;padding:20px;border-radius:6px;margin-bottom:10px;border:1px solid #4B5563;">
                <div style="font-size:24px;font-weight:bold;color:#FFFFFF;">{aaa_pass}/{len(valid_color_results)} ({aaa_percent:.1f}%)</div>
                <div style="font-size:14px;color:#E0E0E0;margin-top:5px;">WCAG AAA (normalny tekst)</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div style="background-color:#262730;padding:20px;border-radius:6px;margin-bottom:10px;border:1px solid #4B5563;">
                <div style="font-size:24px;font-weight:bold;color:#FFFFFF;">{outside_safe_area}/{len(regions_with_analysis)} ({outside_percent:.1f}%)</div>
                <div style="font-size:14px;color:#E0E0E0;margin-top:5px;">Elementy poza strefą bezpieczną</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Wgraj plik graficzny, aby rozpocząć analizę.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("Wystąpił krytyczny błąd:")
        st.exception(e)
