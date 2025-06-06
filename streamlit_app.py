import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn.cluster import KMeans
import os
import time
import gc

# Konfiguracja środowiska
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'  # Optymalizacja pamięci dla PaddleOCR

# Funkcja inicjalizująca OCR - z parametrem ttl i limitami pamięci
@st.cache_resource(show_spinner=False, ttl=300)  # 5 minut czasu życia cache
def load_ocr_reader(languages=['pl']):
    """
    Próbuje załadować czytnik PaddleOCR. Zwraca None w przypadku niepowodzenia.
    """
    try:
        with st.spinner("Ładowanie modeli OCR... To może potrwać chwilę przy pierwszym uruchomieniu."):
            # Importujemy paddleocr wewnątrz funkcji
            from paddleocr import PaddleOCR
            
            # Mapowanie języków do kodów używanych przez PaddleOCR
            lang_map = {
                'en': 'en',
                'pl': 'pl',
                'de': 'german',
                'fr': 'french',
                'es': 'spanish',
                'it': 'italian'
            }
            
            # Wybieramy pierwszy dostępny język lub domyślnie polski
            lang = next((lang_map[l] for l in languages if l in lang_map), 'pl')
            
            # Inicjalizacja PaddleOCR z lekkim modelem
            ocr = PaddleOCR(
                use_angle_cls=True,      # Wykrywanie orientacji tekstu
                lang=lang,               # Język rozpoznawania
                use_gpu=False,           # Brak GPU w Streamlit Cloud
                show_log=False,          # Wyłączenie logów
                use_mp=True,             # Wielowątkowość dla szybszego przetwarzania
                enable_mkldnn=True,      # Optymalizacja CPU
                use_tensorrt=False,      # Wyłączenie TensorRT (nie jest dostępne w Streamlit Cloud)
                cpu_threads=2,           # Mniej wątków dla oszczędności pamięci
                det_db_thresh=0.3,       # Niższy próg detekcji
                det_db_box_thresh=0.5,   # Niższy próg dla bounding boxów
                det_limit_side_len=960   # Ograniczenie rozmiaru obrazu
            )
            return ocr
    except Exception as e:
        st.error(f"Błąd podczas inicjalizacji PaddleOCR: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Funkcja do detekcji tekstu z mechanizmem ponownych prób
def detect_text_paddleocr(image, ocr, max_retries=3):
    """
    Wykrywa tekst na obrazie za pomocą PaddleOCR z mechanizmem ponownych prób.
    """
    # Sprawdzamy czy czytnik OCR jest dostępny
    if ocr is None:
        st.warning("Rozpoznawanie tekstu nie jest dostępne.")
        return []
    
    # Implementacja mechanizmu ponownych prób
    for attempt in range(max_retries):
        try:
            # Konwersja obrazu do formatu wymaganego przez PaddleOCR
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image
            
            # Detekcja tekstu
            with st.spinner(f"Wykrywanie tekstu na obrazie (próba {attempt+1}/{max_retries})..."):
                # Jawne zwolnienie pamięci przed wykrywaniem
                gc.collect()
                
                # Wykonaj detekcję tekstu
                results = ocr.ocr(img_array, cls=True)
                
                # Przetwarzamy wyniki
                text_regions = []
                
                # PaddleOCR zwraca wyniki dla każdej strony (zwykle jednej)
                for page_result in results:
                    if not page_result:  # Pusta strona
                        continue
                        
                    for result in page_result:
                        # PaddleOCR zwraca [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], tekst, pewność
                        box = result[0]
                        text = result[1][0]  # Tekst
                        confidence = result[1][1] * 100  # Pewność w procentach
                        
                        # Konwersja do formatu (x, y, x+w, y+h)
                        x_min = min(point[0] for point in box)
                        y_min = min(point[1] for point in box)
                        x_max = max(point[0] for point in box)
                        y_max = max(point[1] for point in box)
                        
                        text_regions.append({
                            'text': text,
                            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
                            'confidence': confidence
                        })
                
                # Wykonane pomyślnie - zwróć wyniki
                return text_regions
                
        except Exception as e:
            # Jeśli to nie ostatnia próba, spróbuj ponownie
            if attempt < max_retries - 1:
                # Czekaj z opadającym opóźnieniem przed kolejną próbą
                wait_time = 1 * (2 ** attempt)
                st.warning(f"Problem z detekcją tekstu, ponawiam za {wait_time}s...")
                time.sleep(wait_time)
                
                # Spróbuj zwolnić zasoby
                gc.collect()
            else:
                # Wszystkie próby nieudane
                st.error(f"Błąd podczas detekcji tekstu po {max_retries} próbach: {e}")
                import traceback
                st.error(traceback.format_exc())
                return []

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

# ULEPSZONE PODEJŚCIE: Pobieranie i analiza kolorów z próbek z uwzględnieniem proporcji pikseli
def sample_colors_from_region(image, region, num_samples=10):
    """
    Pobiera próbki kolorów z różnych części regionu i analizuje je statystycznie.
    Wykorzystuje zarówno luminancję jak i proporcję pikseli do określenia tła i tekstu.
    """
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
    
    # Ustal próg dominacji – jeśli większy klaster stanowi określony odsetek pikseli, przyjmujemy go jako tło
    total_pixels = len(pixels)
    dominant_threshold = 0.7  # 70%
    ratio_dominant = counts[sorted_indices[0]] / total_pixels
    
    # Określanie tła i tekstu na podstawie zarówno proporcji jak i luminancji
    if ratio_dominant >= dominant_threshold:
        # Jeśli jeden kolor dominuje, przyjmujemy go jako tło niezależnie od jasności
        background_color = color1  # większy klaster to tło
        foreground_color = color2  # mniejszy klaster to tekst
    else:
        # Jeśli nie ma wyraźnej dominacji, porównujemy luminancję
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
        'compliance': compliance,
        'ratio_dominant': ratio_dominant  # Dodajemy informację o proporcji dominującego koloru
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
    Wizualizuje wyniki analizy na obrazie.
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
        st.error(f"Błąd podczas wizualizacji strefy bezpiecznej: {e}")
        st.image(image, caption="Oryginalna grafika", use_container_width=True)

# Funkcja do wyświetlania analizy
def show_analysis_results(regions_with_analysis, margin_h_percent, margin_v_percent, margin_h_px, margin_v_px):
    # Wyświetl legendę
    st.markdown("""
    **Legenda:**
    - **Ciemniejszy obszar** - Strefa marginesu (unikaj umieszczania tam treści)
    - **Jaśniejszy obszar** - Strefa bezpieczna dla treści
    - **Zielona ramka** - Element zgodny z WCAG AAA
    - **Niebieska ramka** - Element zgodny z WCAG AA, ale nie z AAA
    - **Czerwona ramka** - Element niezgodny z WCAG AA
    """)
    
    # Wyświetl szczegóły analizy
    st.subheader("Podsumowanie wyników")
    
    # Style dla kart z danymi
    st.markdown("""
    <style>
    .metric-card {
        background-color: #262730;
        border-radius: 6px;
        padding: 20px;
        margin-bottom: 10px;
        border: 1px solid #4B5563;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #E0E0E0;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    valid_color_results = [r for r in regions_with_analysis if r['colors'] is not None]
    
    aa_pass = sum(1 for r in valid_color_results if r['colors']['compliance']['AA_normal'])
    aa_fail = len(valid_color_results) - aa_pass
    aa_percent = (aa_pass / len(valid_color_results) * 100) if len(valid_color_results) > 0 else 0
    
    aaa_pass = sum(1 for r in valid_color_results if r['colors']['compliance']['AAA_normal'])
    aaa_fail = len(valid_color_results) - aaa_pass
    aaa_percent = (aaa_pass / len(valid_color_results) * 100) if len(valid_color_results) > 0 else 0
    
    outside_safe_area = sum(1 for r in regions_with_analysis if not r['edge']['is_inside_safe_area'])
    outside_percent = (outside_safe_area / len(regions_with_analysis) * 100) if len(regions_with_analysis) > 0 else 0
    
    # Podsumowanie
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{aa_pass}/{len(valid_color_results)} ({aa_percent:.1f}%)</div>
            <div class="metric-label">WCAG AA (normalny tekst)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{aaa_pass}/{len(valid_color_results)} ({aaa_percent:.1f}%)</div>
            <div class="metric-label">WCAG AAA (normalny tekst)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{outside_safe_area}/{len(regions_with_analysis)} ({outside_percent:.1f}%)</div>
            <div class="metric-label">Elementy poza strefą bezpieczną</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabela ze szczegółami
    st.subheader("Szczegółowe wyniki")
    
    # Zakładki dla przejrzystości
    tabs = st.tabs(["Wszystkie elementy", "Problemy z kontrastem", "Elementy poza strefą bezpieczną"])
    
    # Zakładka wszystkich elementów
    with tabs[0]:
        for i, region_data in enumerate(regions_with_analysis):
            region = region_data['region']
            color_data = region_data['colors']
            edge_data = region_data['edge']
            
            status_icon = "✅" if (color_data and color_data['compliance']['AA_normal'] and edge_data['is_inside_safe_area']) else "❌"
            
            # Usunięto argument key z expander, który powodował problemy
            with st.expander(f"{status_icon} Element {i+1}: {region['text']}"):
                # Dodaj opcję ręcznej korekty kolorów
                if color_data:
                    # Dodaj informację o proporcji pikseli jeśli dostępna
                    ratio_info = ""
                    if 'ratio_dominant' in color_data:
                        ratio_percent = color_data['ratio_dominant'] * 100
                        ratio_info = f" (dominujący kolor: {ratio_percent:.1f}%)"
                        
                    # Kolorowa ramka dla wartości kontrastu
                    contrast_color = "green" if color_data['compliance']['AAA_normal'] else ("blue" if color_data['compliance']['AA_normal'] else "red")
                    st.markdown(f"""
                    <div style="padding: 10px; border-left: 4px solid {contrast_color}; background-color: #262730; margin-bottom: 10px;">
                        <strong style="color: white;">Współczynnik kontrastu:</strong> <span style="font-size: 16px; font-weight: bold; color: white;">{color_data['contrast_ratio']:.2f}</span>{ratio_info}
                        <br><span style="font-size: 13px; color: #E0E0E0;">Minimum dla WCAG AA: 4.5, dla WCAG AAA: 7.0</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Tabela zgodności
                    st.markdown("""
                    <table style="width:100%; border-collapse: collapse; margin-bottom: 15px; color: white;">
                        <tr style="background-color: #333;">
                            <th style="padding: 8px; text-align: left; border: 1px solid #555;">Poziom WCAG</th>
                            <th style="padding: 8px; text-align: left; border: 1px solid #555;">Normalny tekst</th>
                            <th style="padding: 8px; text-align: left; border: 1px solid #555;">Duży tekst</th>
                        </tr>
                    """, unsafe_allow_html=True)
                    
                    aa_normal = "✅" if color_data['compliance']['AA_normal'] else "❌"
                    aa_large = "✅" if color_data['compliance']['AA_large'] else "❌"
                    aaa_normal = "✅" if color_data['compliance']['AAA_normal'] else "❌"
                    aaa_large = "✅" if color_data['compliance']['AAA_large'] else "❌"
                    
                    st.markdown(f"""
                        <tr>
                            <td style="padding: 8px; border: 1px solid #555;">WCAG AA</td>
                            <td style="padding: 8px; border: 1px solid #555;">{aa_normal}</td>
                            <td style="padding: 8px; border: 1px solid #555;">{aa_large}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #555;">WCAG AAA</td>
                            <td style="padding: 8px; border: 1px solid #555;">{aaa_normal}</td>
                            <td style="padding: 8px; border: 1px solid #555;">{aaa_large}</td>
                        </tr>
                    </table>
                    """, unsafe_allow_html=True)
                    
                    # Wizualizacja wykrytych kolorów
                    st.markdown("<div style='color: white;'><strong>Wykryte kolory:</strong></div>", unsafe_allow_html=True)
                    
                    fg_color = color_data['foreground']
                    bg_color = color_data['background']
                    # Konwersja z RGB na HEX
                    fg_hex = rgb2hex([c/255 for c in fg_color])
                    bg_hex = rgb2hex([c/255 for c in bg_color])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"<div style='color: white;'>Kolor tekstu: {fg_hex}</div>", unsafe_allow_html=True)
                        st.markdown(f'<div style="background-color:{fg_hex};width:50px;height:50px;border:1px solid #555;"></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"<div style='color: white;'>Kolor tła: {bg_hex}</div>", unsafe_allow_html=True)
                        st.markdown(f'<div style="background-color:{bg_hex};width:50px;height:50px;border:1px solid #555;"></div>', unsafe_allow_html=True)
                    
                    # Ręczna korekta kolorów
                    st.markdown("<div style='color: white;'><strong>Ręczna korekta kolorów:</strong></div>", unsafe_allow_html=True)
                    
                    # Użyj color pickerów do ręcznej korekty - każdy z unikalnym kluczem
                    col1, col2 = st.columns(2)
                    with col1:
                        manual_fg = st.color_picker("Skoryguj kolor tekstu", fg_hex, key=f"fg_{i}")
                    with col2:
                        manual_bg = st.color_picker("Skoryguj kolor tła", bg_hex, key=f"bg_{i}")
                    
                    # Oblicz nowy kontrast dla ręcznie skorygowanych kolorów
                    # Konwersja z HEX do RGB
                    def hex_to_rgb(hex_color):
                        hex_color = hex_color.lstrip('#')
                        return [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
                    
                    manual_fg_rgb = hex_to_rgb(manual_fg)
                    manual_bg_rgb = hex_to_rgb(manual_bg)
                    
                    manual_contrast_ratio = calculate_contrast_ratio(manual_fg_rgb, manual_bg_rgb)
                    manual_compliance = check_wcag_compliance(manual_contrast_ratio)
                    
                    if manual_fg != fg_hex or manual_bg != bg_hex:
                        st.markdown(f"""
                        <div style="padding: 10px; background-color: #262730; margin: 15px 0; border: 1px solid #555;">
                            <strong style="color: white;">Skorygowany kontrast:</strong> <span style="font-size: 16px; font-weight: bold; color: white;">{manual_contrast_ratio:.2f}</span>
                            <br><span style="color: white;">WCAG AA (normalny tekst): {'✅' if manual_compliance['AA_normal'] else '❌'}</span>
                            <br><span style="color: white;">WCAG AAA (normalny tekst): {'✅' if manual_compliance['AAA_normal'] else '❌'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color: white;'><strong>Nie udało się wykryć kolorów dla tego elementu.</strong></div>", unsafe_allow_html=True)
                
                # Status bezpiecznego obszaru
                safe_status = "✅" if edge_data['is_inside_safe_area'] else "❌"
                safe_color = "green" if edge_data['is_inside_safe_area'] else "red"
                st.markdown(f"""
                <div style="padding: 10px; border-left: 4px solid {safe_color}; background-color: #262730; margin: 15px 0;">
                    <strong style="color: white;">W strefie bezpiecznej:</strong> {safe_status}
                    <br><span style="font-size: 13px; color: #E0E0E0;">Marginesy: poziomy {margin_h_percent}% ({margin_h_px}px), pionowy {margin_v_percent}% ({margin_v_px}px)</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Odległości w pikselach
                st.markdown("<div style='color: white;'><strong>Odległości od krawędzi (w pikselach):</strong></div>", unsafe_allow_html=True)
                distances = edge_data['distances']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lewa", f"{distances['left']} px", delta=f"{distances['left']-margin_h_px} px", delta_color="normal")
                with col2:
                    st.metric("Prawa", f"{distances['right']} px", delta=f"{distances['right']-margin_h_px} px", delta_color="normal")
                with col3:
                    st.metric("Górna", f"{distances['top']} px", delta=f"{distances['top']-margin_v_px} px", delta_color="normal")
                with col4:
                    st.metric("Dolna", f"{distances['bottom']} px", delta=f"{distances['bottom']-margin_v_px} px", delta_color="normal")
                
                # Odległości procentowe
                st.markdown("<div style='color: white;'><strong>Odległości od krawędzi (w procentach):</strong></div>", unsafe_allow_html=True)
                percentages = edge_data['percentages']
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lewa", f"{percentages['left']:.1f}%", delta=f"{percentages['left']-margin_h_percent:.1f}%", delta_color="normal")
                with col2:
                    st.metric("Prawa", f"{percentages['right']:.1f}%", delta=f"{percentages['right']-margin_h_percent:.1f}%", delta_color="normal")
                with col3:
                    st.metric("Górna", f"{percentages['top']:.1f}%", delta=f"{percentages['top']-margin_v_percent:.1f}%", delta_color="normal")
                with col4:
                    st.metric("Dolna", f"{percentages['bottom']:.1f}%", delta=f"{percentages['bottom']-margin_v_percent:.1f}%", delta_color="normal")
    
    # Zakładka problemów z kontrastem
    with tabs[1]:
        problematic_contrast = [r for r in regions_with_analysis if r['colors'] is None or not r['colors']['compliance']['AA_normal']]
        
        if not problematic_contrast:
            st.success("Nie znaleziono problemów z kontrastem - wszystkie elementy spełniają wymagania WCAG AA.")
        else:
            st.error(f"Znaleziono {len(problematic_contrast)} element(ów) z problemami kontrastu:")
            
            for i, region_data in enumerate(problematic_contrast):
                region = region_data['region']
                color_data = region_data['colors']
                
                with st.expander(f"Problem {i+1}: {region['text']}"):
                    if color_data:
                        st.markdown(f"""<div style='color: white;'><strong>Współczynnik kontrastu:</strong> {color_data['contrast_ratio']:.2f} (Wymagane minimum: 4.5)</div>""", unsafe_allow_html=True)
                        
                        # Sugestia poprawy
                        st.markdown("<div style='color: white;'><strong>Sugestia poprawy:</strong></div>", unsafe_allow_html=True)
                        st.markdown("<div style='color: white;'>Zwiększ kontrast między tekstem a tłem, stosując ciemniejszy tekst na jasnym tle lub jaśniejszy tekst na ciemnym tle.</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color: white;'><strong>Nie udało się wykryć kolorów dla tego elementu.</strong></div>", unsafe_allow_html=True)
    
    # Zakładka problemów z położeniem
    with tabs[2]:
        outside_elements = [r for r in regions_with_analysis if not r['edge']['is_inside_safe_area']]
        
        if not outside_elements:
            st.success("Nie znaleziono elementów poza strefą bezpieczną.")
        else:
            st.error(f"Znaleziono {len(outside_elements)} element(ów) poza strefą bezpieczną:")
            
            for i, region_data in enumerate(outside_elements):
                region = region_data['region']
                edge_data = region_data['edge']
                
                with st.expander(f"Problem {i+1}: {region['text']}"):
                    distances = edge_data['distances']
                    
                    # Znajdź problematyczne krawędzie
                    problem_edges = []
                    if distances['left'] < margin_h_px:
                        problem_edges.append(f"Lewa ({distances['left']} px, minimum: {margin_h_px} px)")
                    if distances['right'] < margin_h_px:
                        problem_edges.append(f"Prawa ({distances['right']} px, minimum: {margin_h_px} px)")
                    if distances['top'] < margin_v_px:
                        problem_edges.append(f"Górna ({distances['top']} px, minimum: {margin_v_px} px)")
                    if distances['bottom'] < margin_v_px:
                        problem_edges.append(f"Dolna ({distances['bottom']} px, minimum: {margin_v_px} px)")
                    
                    st.markdown(f"""<div style='color: white;'><strong>Zbyt mała odległość od krawędzi:</strong> {', '.join(problem_edges)}</div>""", unsafe_allow_html=True)
                    
                    # Sugestia poprawy
                    st.markdown("<div style='color: white;'><strong>Sugestia poprawy:</strong></div>", unsafe_allow_html=True)
                    st.markdown(f"""<div style='color: white;'>Przesuń element aby znajdował się całkowicie w strefie bezpiecznej (marginesy: poziomy {margin_h_px}px, pionowy {margin_v_px}px).</div>""", unsafe_allow_html=True)

# Główna aplikacja Streamlit
def main():
    st.title("Analiza Dostępności Grafiki - WCAG")
    
    # Sidebar z opcjami
    st.sidebar.header("Opcje analizy")
    
    # Opcje języka dla OCR - domyślnie tylko polski
    languages = st.sidebar.multiselect(
        "Języki OCR",
        options=["pl", "en", "de", "fr", "es", "it"],
        default=["pl"],  # Domyślnie tylko polski
        key="ocr_languages"
    )
    
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
    
    # Inicjalizacja zmiennych stanu sesji jeśli nie istnieją
    if 'previous_file' not in st.session_state:
        st.session_state.previous_file = None
        st.session_state.previous_margins = (0, 0)
        st.session_state.detected_regions = None
        st.session_state.cv_image = None
    
    if uploaded_file is not None:
        # Sprawdź, czy plik został zmieniony
        file_changed = st.session_state.previous_file is None or uploaded_file.name != st.session_state.previous_file
        
        # Sprawdź, czy marginesy zostały zmienione
        margins_changed = not file_changed and (st.session_state.previous_margins != (margin_h_percent, margin_v_percent))
        
        # Aktualizacja stanu sesji
        st.session_state.previous_file = uploaded_file.name
        st.session_state.previous_margins = (margin_h_percent, margin_v_percent)
        
        # Wczytaj obraz tylko jeśli się zmienił
        if file_changed:
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
            
            # Zapisz obraz w stanie sesji
            st.session_state.cv_image = cv_image
        else:
            # Użyj zapisanego obrazu
            cv_image = st.session_state.cv_image
            pil_image = Image.open(uploaded_file)
        
        # Wyświetl oryginalny obraz
        st.image(pil_image, caption="Wgrana grafika", use_container_width=True)
        
        # Oblicz marginesy w pikselach na podstawie wymiarów obrazu
        height, width = cv_image.shape[:2]
        margin_h_px = int((width * margin_h_percent) / 100)
        margin_v_px = int((height * margin_v_percent) / 100)
        
        # Wyświetl informację o marginesach
        st.info(f"Bezpieczny obszar: margines poziomy {margin_h_percent}% ({margin_h_px}px), margines pionowy {margin_v_percent}% ({margin_v_px}px)")
        
        # Automatyczne wykrywanie tekstu - tylko gdy plik się zmienił
        if file_changed:
            with st.spinner("Inicjalizacja PaddleOCR i wykrywanie tekstu..."):
                # Inicjalizacja czytnika OCR
                reader = load_ocr_reader(languages)
                
                if reader:
                    # Wykrywanie tekstu
                    regions = detect_text_paddleocr(cv_image, reader, max_retries=3)
                    st.session_state.detected_regions = regions
                    
                    if not regions:
                        st.warning("Nie wykryto tekstu. Sprawdź czy obraz zawiera wyraźny tekst.")
                        # Pokaż tylko strefy bezpieczne jeśli nie wykryto tekstu
                        show_safety_margin(cv_image, margin_h_px, margin_v_px)
                else:
                    st.error("Nie udało się zainicjalizować PaddleOCR.")
                    regions = []
                    st.session_state.detected_regions = []
                    # Pokaż tylko strefy bezpieczne jeśli nie ma OCR
                    show_safety_margin(cv_image, margin_h_px, margin_v_px)
        else:
            # Użyj zapisanych regionów
            regions = st.session_state.detected_regions
        
        # Sprawdzanie czy mamy regiony tekstu do analizy
        if regions:
            with st.spinner("Analiza obszarów..."):
                regions_with_analysis = []
                
                for region in regions:
                    # Analizuj kolory używając ulepszonego podejścia z proporcją pikseli
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
                
                # Reszta kodu do wyświetlania wyników
                show_analysis_results(regions_with_analysis, margin_h_percent, margin_v_percent, margin_h_px, margin_v_px)

if __name__ == "__main__":
    main()