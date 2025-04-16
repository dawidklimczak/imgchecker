import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import easyocr

# Sprawdzenie czy CUDA jest dostępne (dla szybszego OCR)
# Używamy zmiennej środowiskowej, aby uniknąć problemów z śledzeniem modułów
import os
os.environ["PYTORCH_JIT"] = "0"  # Wyłączenie JIT, które może powodować problemy
GPU_AVAILABLE = False  # Domyślnie przyjmujemy, że GPU nie jest dostępne na Streamlit Cloud

# Funkcja do obliczenia współczynnika kontrastu WCAG
def calculate_contrast_ratio(color1, color2):
    # Konwersja kolorów RGB na luminancję
    def get_luminance(rgb):
        rgb = [c/255 for c in rgb]
        rgb = [c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    
    luminance1 = get_luminance(color1)
    luminance2 = get_luminance(color2)
    
    # Obliczenie współczynnika kontrastu
    if luminance1 > luminance2:
        return (luminance1 + 0.05) / (luminance2 + 0.05)
    else:
        return (luminance2 + 0.05) / (luminance1 + 0.05)

# Funkcja do oceny zgodności z WCAG AA i AAA
def check_wcag_compliance(contrast_ratio):
    aa_normal_text = contrast_ratio >= 4.5
    aa_large_text = contrast_ratio >= 3.0
    aaa_normal_text = contrast_ratio >= 7.0
    aaa_large_text = contrast_ratio >= 4.5
    
    return {
        'AA_normal': aa_normal_text,
        'AA_large': aa_large_text,
        'AAA_normal': aaa_normal_text,
        'AAA_large': aaa_large_text
    }

# Inicjalizacja czytnika EasyOCR z obsługą awarii
@st.cache_resource
def load_ocr_reader(languages=['pl', 'en']):
    if not EASYOCR_AVAILABLE:
        if DEBUG_MODE:
            st.warning("EasyOCR nie jest dostępny. Funkcja rozpoznawania tekstu nie będzie działać.")
        return None
        
    try:
        # Próbujemy zainicjalizować czytnik z minimalnymi parametrami
        reader = easyocr.Reader(
            languages, 
            gpu=False,            # Wyłączenie GPU
            verbose=False,        # Wyłączenie komunikatów
            quantize=True,        # Redukcja rozmiaru modeli
            cudnn_benchmark=False # Wyłączenie optymalizacji CUDA
        )
        return reader
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Nie udało się zainicjalizować EasyOCR: {e}")
        return None

# Funkcja do detekcji tekstu za pomocą EasyOCR lub obsługi błędów
def detect_text_easyocr(image, reader):
    # Sprawdzamy czy czytnik EasyOCR jest dostępny
    if reader is None:
        if DEBUG_MODE:
            st.warning("Rozpoznawanie tekstu nie jest dostępne.")
        return []
    
    try:
        # Konwersja obrazu do formatu wymaganego przez EasyOCR
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            img_array = image
        
        # Detekcja tekstu z obsługą błędów
        text_regions = []
        
        # Próbujemy wykonać detekcję z minimalnymi parametrami
        results = reader.readtext(
            img_array,
            detail=1,           # Pełne szczegóły wykrytego tekstu
            paragraph=True,     # Grupowanie tekstu w paragrafy (szybsze)
            height_ths=0.5,     # Wyższy próg dla łączenia linii (mniej wyników)
            width_ths=0.5,      # Wyższy próg dla łączenia znaków (mniej wyników)
            decoder='greedy'    # Najszybszy dekoder
        )
        
        # Przetwarzamy wyniki
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
        
        return text_regions
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Błąd podczas detekcji tekstu: {e}")
        return []

# NOWE PODEJŚCIE: Pobieranie i analiza kolorów z próbek
def sample_colors_from_region(image, region, num_samples=10):
    """Pobiera próbki kolorów z różnych części regionu i analizuje je statystycznie."""
    x1, y1, x2, y2 = region['bbox']
    
    # Upewnij się, że współrzędne są w granicach obrazu
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    # Pobierz region
    region_img = image[y1:y2, x1:x2]
    
    if region_img.size == 0:
        return None
    
    # Konwersja do RGB (z BGR)
    if len(region_img.shape) == 3:
        region_rgb = cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB)
    else:
        # Jeśli obraz jest w skali szarości, konwertuj do RGB
        region_rgb = cv2.cvtColor(region_img, cv2.COLOR_GRAY2RGB)
    
    # Histogram kolorów dla znalezienia dominujących kolorów
    pixels = region_rgb.reshape(-1, 3)
    
    # Proste podejście klastrujące - znajdź dwa główne klastry kolorów
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, random_state=42)
    km.fit(pixels)
    
    # Centra klastrów to dominujące kolory
    centers = km.cluster_centers_.astype(int)
    
    # Etykiety klastrów
    labels = km.labels_
    
    # Ilość pikseli w każdym klastrze
    counts = np.bincount(labels)
    
    # Sortowanie według liczebności
    sorted_indices = np.argsort(counts)[::-1]
    
    # Dwa najbardziej dominujące kolory
    color1 = centers[sorted_indices[0]]
    color2 = centers[sorted_indices[1]]
    
    # Oblicz luminancję obu kolorów
    def get_luminance(rgb):
        rgb = [c/255 for c in rgb]
        rgb = [c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4 for c in rgb]
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    
    # Przypisz jaśniejszy kolor jako tło, a ciemniejszy jako tekst
    # Typowo tekst jest ciemniejszy niż tło
    if lum1 > lum2:
        background_color = color1
        foreground_color = color2
    else:
        background_color = color2
        foreground_color = color1
    
    # Oblicz współczynnik kontrastu
    contrast_ratio = calculate_contrast_ratio(foreground_color, background_color)
    
    # Sprawdź zgodność z WCAG
    compliance = check_wcag_compliance(contrast_ratio)
    
    # Dodaj próbki koloru do wizualizacji
    samples = {
        'foreground': foreground_color,
        'background': background_color,
        'contrast_ratio': contrast_ratio,
        'compliance': compliance
    }
    
    return samples

# Funkcja do sprawdzenia odległości od krawędzi
def check_edge_distance(image_shape, region, margin_h_px, margin_v_px):
    try:
        x1, y1, x2, y2 = region['bbox']
        height, width = image_shape[:2]
        
        # Oblicz współrzędne bezpiecznego obszaru
        safe_x1 = margin_h_px
        safe_y1 = margin_v_px
        safe_x2 = width - margin_h_px
        safe_y2 = height - margin_v_px
        
        # Sprawdź, czy element jest całkowicie w bezpiecznym obszarze
        is_inside = (x1 >= safe_x1 and y1 >= safe_y1 and x2 <= safe_x2 and y2 <= safe_y2)
        
        # Oblicz odległości od krawędzi
        distance_left = x1
        distance_right = width - x2
        distance_top = y1
        distance_bottom = height - y2
        
        # Oblicz procentowe odległości od krawędzi
        percent_left = (distance_left / width) * 100
        percent_right = (distance_right / width) * 100
        percent_top = (distance_top / height) * 100
        percent_bottom = (distance_bottom / height) * 100
        
        return {
            'is_inside_safe_area': is_inside,
            'distances': {
                'left': distance_left,
                'right': distance_right,
                'top': distance_top,
                'bottom': distance_bottom
            },
            'percentages': {
                'left': percent_left,
                'right': percent_right,
                'top': percent_top,
                'bottom': percent_bottom
            }
        }
    except Exception as e:
        st.error(f"Błąd podczas sprawdzania odległości od krawędzi: {e}")
        return {
            'is_inside_safe_area': False,
            'distances': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0},
            'percentages': {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
        }

# Funkcja do wizualizacji wyników
def visualize_results(original_image, regions_with_analysis, margin_h_px, margin_v_px):
    """
    Nowe podejście do wizualizacji: tworzymy nową warstwę z oznaczeniami, 
    ale zachowujemy oryginalne kolory obrazu.
    """
    try:
        # Stwórz kopię obrazu do wizualizacji
        # Konwertuj BGR do RGB dla poprawnego wyświetlania
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            vis_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            vis_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Stwórz warstwę z oznaczeniami jako osobny obraz
        height, width = original_image.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA dla przezroczystości
        
        # Współrzędne bezpiecznego obszaru
        safe_x1 = margin_h_px
        safe_y1 = margin_v_px
        safe_x2 = width - margin_h_px
        safe_y2 = height - margin_v_px
        
        # Dodaj półprzezroczysty obszar marginesu
        for y in range(height):
            for x in range(width):
                if (x < safe_x1 or x > safe_x2 or y < safe_y1 or y > safe_y2):
                    overlay[y, x] = [0, 0, 0, 128]  # Czarny z przezroczystością 50%
        
        # Dodaj ramkę wokół bezpiecznego obszaru
        # Górna linia
        overlay[safe_y1-1:safe_y1+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        # Dolna linia
        overlay[safe_y2-1:safe_y2+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        # Lewa linia
        overlay[safe_y1-1:safe_y2+1, safe_x1-1:safe_x1+1] = [255, 255, 255, 200]
        # Prawa linia
        overlay[safe_y1-1:safe_y2+1, safe_x2-1:safe_x2+1] = [255, 255, 255, 200]
        
        # Dodaj etykietę bezpiecznego obszaru
        # OpenCV nie obsługuje przezroczystości, więc dodamy tekst bezpośrednio do finalnego obrazu później
        
        # Zaznacz regiony tekstowe
        for region_data in regions_with_analysis:
            region = region_data['region']
            color_data = region_data['colors']
            edge_data = region_data['edge']
            
            if not color_data:
                continue
                
            x1, y1, x2, y2 = region['bbox']
            compliance = color_data['compliance']
            
            # Ustal kolor ramki na podstawie zgodności z WCAG (w formacie RGBA)
            if compliance['AAA_normal']:
                box_color = [0, 255, 0, 200]  # Zielony - zgodny z AAA
            elif compliance['AA_normal']:
                box_color = [0, 0, 255, 200]  # Niebieski - zgodny z AA, ale nie z AAA
            else:
                box_color = [255, 0, 0, 200]  # Czerwony - niezgodny z AA
            
            # Narysuj ramkę wokół regionu tekstu
            # Górna linia
            overlay[y1-1:y1+1, x1-1:x2+1] = box_color
            # Dolna linia
            overlay[y2-1:y2+1, x1-1:x2+1] = box_color
            # Lewa linia
            overlay[y1-1:y2+1, x1-1:x1+1] = box_color
            # Prawa linia
            overlay[y1-1:y2+1, x2-1:x2+1] = box_color
            
            # Dodaj informację o współczynniku kontrastu
            # OpenCV nie obsługuje przezroczystości, więc dodamy tekst bezpośrednio do finalnego obrazu później
        
        # Połącz warstwę oznaczeń z oryginalnym obrazem
        # Konwertuj overlay do formatu PIL
        overlay_pil = Image.fromarray(overlay)
        original_pil = Image.fromarray(vis_image_rgb)
        
        # Nałóż overlay na oryginalny obraz
        result = Image.alpha_composite(original_pil.convert("RGBA"), overlay_pil)
        
        # Konwertuj z powrotem do formatu numpy dla dalszego przetwarzania
        result_array = np.array(result)
        
        # Dodaj teksty (nie można ich dodać w warstwie RGBA bezpośrednio przez OpenCV)
        # Konwertuj do RGB (bez kanału alpha) dla OpenCV
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_RGBA2RGB)
        
        # Dodaj etykietę bezpiecznego obszaru
        cv2.putText(result_rgb, "Strefa bezpieczna dla treści", (safe_x1 + 10, safe_y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Dodaj informacje o współczynnikach kontrastu
        for region_data in regions_with_analysis:
            region = region_data['region']
            color_data = region_data['colors']
            
            if not color_data:
                continue
                
            x1, y1, x2, y2 = region['bbox']
            contrast_text = f"{color_data['contrast_ratio']:.2f}"
            
            # Wybierz kolor tekstu dla kontrastu
            if color_data['compliance']['AAA_normal']:
                text_color = (0, 255, 0)  # Zielony
            elif color_data['compliance']['AA_normal']:
                text_color = (0, 0, 255)  # Niebieski
            else:
                text_color = (255, 0, 0)  # Czerwony
            
            cv2.putText(result_rgb, contrast_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        return result_rgb
    except Exception as e:
        st.error(f"Błąd podczas wizualizacji wyników: {e}")
        return original_image

# Główna aplikacja Streamlit
def main():
    st.title("Analiza Dostępności Grafiki - WCAG")
    
    # Sidebar z opcjami
    st.sidebar.header("Opcje analizy")
    
    # Opcje języka dla OCR (tylko jeśli EasyOCR jest dostępny)
    if EASYOCR_AVAILABLE:
        languages = st.sidebar.multiselect(
            "Języki OCR",
            options=["pl", "en", "de", "fr", "es", "it"],
            default=["pl", "en"],
            key="ocr_languages"
        )
    else:
        languages = ["pl", "en"]  # Domyślne języki
        if DEBUG_MODE:
            st.sidebar.warning("EasyOCR nie jest dostępny. Wybór języków został wyłączony.")
    
    
    # Ustawienia marginesów
    st.sidebar.subheader("Bezpieczny obszar")
    
    # Domyślny margines 10% dla szerokości i wysokości
    margin_h_percent = st.sidebar.slider(
        "Margines poziomy (%)",
        1, 25, 10,
        help="Minimalny odstęp od lewej i prawej krawędzi jako procent szerokości obrazu",
        key="margin_h_slider"
    )
    
    margin_v_percent = st.sidebar.slider(
        "Margines pionowy (%)",
        1, 25, 10,
        help="Minimalny odstęp od górnej i dolnej krawędzi jako procent wysokości obrazu",
        key="margin_v_slider"
    )
    
    # Upload pliku
    uploaded_file = st.file_uploader("Wybierz plik graficzny", type=["png", "jpg", "jpeg"], key="file_uploader")
    
    if uploaded_file is not None:
        # Wczytaj obraz
        pil_image = Image.open(uploaded_file)
        
        # Konwersja do formatu OpenCV (BGR)
        cv_image = np.array(pil_image)
        if len(cv_image.shape) == 2:  # Jeśli obraz jest w skali szarości
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # Jeśli obraz ma kanał alfa (RGBA)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
        elif len(cv_image.shape) == 3 and cv_image.shape[2] == 3:  # RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        # Wyświetl oryginalny obraz
        st.image(pil_image, caption="Wgrana grafika", use_container_width=True)
        
        # Oblicz marginesy w pikselach na podstawie wymiarów obrazu
        height, width = cv_image.shape[:2]
        margin_h_px = int((width * margin_h_percent) / 100)
        margin_v_px = int((height * margin_v_percent) / 100)
        
        # Wyświetl informację o marginesach
        st.info(f"Bezpieczny obszar: margines poziomy {margin_h_percent}% ({margin_h_px}px), margines pionowy {margin_v_percent}% ({margin_v_px}px)")
        
        regions = []
        
        # Automatyczne wykrywanie tekstu
        reader = None
        regions = []
        
        # Inicjalizacja czytnika OCR tylko jeśli EasyOCR jest dostępny
        if EASYOCR_AVAILABLE:
            with st.spinner("Inicjalizacja EasyOCR i wykrywanie tekstu..."):
                try:
                    reader = load_ocr_reader(languages)
                    
                    if reader:
                        # Wykrywanie tekstu
                        regions = detect_text_easyocr(cv_image, reader)
                        
                        if not regions:
                            st.warning("Nie wykryto tekstu. Sprawdź czy obraz zawiera wyraźny tekst.")
                    elif DEBUG_MODE:
                        st.error("Nie udało się zainicjalizować EasyOCR.")
                except Exception as e:
                    if DEBUG_MODE:
                        st.error(f"Błąd podczas korzystania z EasyOCR: {e}")
        elif DEBUG_MODE:
            st.warning("EasyOCR nie jest dostępny. Funkcja rozpoznawania tekstu jest wyłączona.")
        
        # Sprawdzanie czy mamy regiony tekstu do analizy
        if regions:
            with st.spinner("Analiza obszarów..."):
                regions_with_analysis = []
                
                for region in regions:
                    # Analizuj kolory używając nowego podejścia
                    color_data = sample_colors_from_region(cv_image, region)
                    
                    # Sprawdź odległość od krawędzi
                    edge_data = check_edge_distance(cv_image.shape, region, margin_h_px, margin_v_px)
                    
                    regions_with_analysis.append({
                        'region': region,
                        'colors': color_data,
                        'edge': edge_data
                    })
                
                # Wizualizuj wyniki z bezpiecznym obszarem i nowym podejściem do kolorów
                result_image = visualize_results(cv_image, regions_with_analysis, margin_h_px, margin_v_px)
                
                # Wyświetl wyniki
                st.subheader("Wyniki analizy")
                st.image(result_image, caption="Analiza dostępności", use_container_width=True)
                
                # Reszta kodu do wyświetlania wyników...
                show_analysis_results(regions_with_analysis, margin_h_percent, margin_v_percent, margin_h_px, margin_v_px)
        else:
            # Jeśli nie mamy regionów tekstu, ale chcemy nadal pokazać margines bezpieczeństwa
            show_safety_margin(cv_image, margin_h_px, margin_v_px)
                
# Nowa funkcja do wyświetlania tylko marginesu bezpieczeństwa
def show_safety_margin(image, margin_h_px, margin_v_px):
    """Wyświetla obraz z zaznaczonym marginesem bezpieczeństwa, bez analizy tekstu."""
    try:
        # Konwertuj BGR do RGB dla poprawnego wyświetlania
        if len(image.shape) == 3 and image.shape[2] == 3:
            vis_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            vis_image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Stwórz warstwę z oznaczeniami jako osobny obraz
        height, width = image.shape[:2]
        overlay = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA dla przezroczystości
        
        # Współrzędne bezpiecznego obszaru
        safe_x1 = margin_h_px
        safe_y1 = margin_v_px
        safe_x2 = width - margin_h_px
        safe_y2 = height - margin_v_px
        
        # Dodaj półprzezroczysty obszar marginesu
        for y in range(height):
            for x in range(width):
                if (x < safe_x1 or x > safe_x2 or y < safe_y1 or y > safe_y2):
                    overlay[y, x] = [0, 0, 0, 128]  # Czarny z przezroczystością 50%
        
        # Dodaj ramkę wokół bezpiecznego obszaru
        # Górna linia
        overlay[safe_y1-1:safe_y1+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        # Dolna linia
        overlay[safe_y2-1:safe_y2+1, safe_x1-1:safe_x2+1] = [255, 255, 255, 200]
        # Lewa linia
        overlay[safe_y1-1:safe_y2+1, safe_x1-1:safe_x1+1] = [255, 255, 255, 200]
        # Prawa linia
        overlay[safe_y1-1:safe_y2+1, safe_x2-1:safe_x2+1] = [255, 255, 255, 200]
        
        # Połącz warstwę oznaczeń z oryginalnym obrazem
        # Konwertuj overlay do formatu PIL
        overlay_pil = Image.fromarray(overlay)
        original_pil = Image.fromarray(vis_image_rgb)
        
        # Nałóż overlay na oryginalny obraz
        result = Image.alpha_composite(original_pil.convert("RGBA"), overlay_pil)
        
        # Konwertuj z powrotem do formatu numpy
        result_array = np.array(result)
        result_rgb = cv2.cvtColor(result_array, cv2.COLOR_RGBA2RGB)
        
        # Dodaj etykietę bezpiecznego obszaru
        cv2.putText(result_rgb, "Strefa bezpieczna dla treści", (safe_x1 + 10, safe_y1 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Wyświetl wyniki
        st.subheader("Strefa bezpieczna")
        st.image(result_rgb, caption="Strefa bezpieczna dla treści", use_container_width=True)
        
        # Wyświetl legendę
        st.markdown("""
        **Legenda:**
        - **Ciemniejszy obszar** - Strefa marginesu (unikaj umieszczania tam treści)
        - **Jaśniejszy obszar** - Strefa bezpieczna dla treści
        """)
        
        # Wyświetl informację o braku wykrytego tekstu
        st.warning("Nie znaleziono tekstu na obrazie lub funkcja wykrywania tekstu jest niedostępna.")
        
    except Exception as e:
        if DEBUG_MODE:
            st.error(f"Błąd podczas wizualizacji strefy bezpiecznej: {e}")
        st.image(image, caption="Oryginalna grafika", use_container_width=True)

if __name__ == "__main__":
    main()

    