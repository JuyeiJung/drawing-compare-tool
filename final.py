import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
import tempfile
import os
import random
import json
import fitz  # PyMuPDF

# -------------------------
# 페이지 설정
# -------------------------
st.set_page_config(page_title="도면 비교 도구", layout="wide")
st.title("도면 비교 도구")

# -------------------------
# 탭 생성
# -------------------------
tab1, tab2 = st.tabs(["🔍 도면 비교 (차이점 강조)", "📊 도면 겹치기 (오버레이)"])

# -------------------------
# 공통 함수들
# -------------------------
def pdf_to_image(pdf_file, page_num=0, dpi=300):
    """PDF 파일을 이미지로 변환 (PyMuPDF 사용)"""
    try:
        # 파일 포인터를 처음으로 되돌리기
        pdf_file.seek(0)
        
        # PDF 데이터 읽기
        pdf_data = pdf_file.read()
        
        # 데이터가 비어있는지 확인
        if not pdf_data:
            raise ValueError("PDF 파일이 비어있습니다")
        
        # PyMuPDF로 PDF 열기
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        # 페이지 수 확인
        if page_num >= pdf_document.page_count:
            page_num = 0
        
        # 페이지 선택
        page = pdf_document[page_num]
        
        # 해상도 설정 (DPI)
        zoom = dpi / 72  # 72 DPI가 기본
        mat = fitz.Matrix(zoom, zoom)
        
        # 이미지로 렌더링
        pix = page.get_pixmap(matrix=mat)
        
        # PIL Image로 변환
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # OpenCV 형식(BGR)으로 변환
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        total_pages = pdf_document.page_count
        pdf_document.close()
        
        return img_bgr, total_pages
    except Exception as e:
        raise Exception(f"PDF 변환 실패: {str(e)}")

def load_file(uploaded_file, page_num=0):
    """PDF 또는 이미지 파일을 OpenCV 이미지로 로드"""
    try:
        # 파일 포인터를 처음으로 되돌리기
        uploaded_file.seek(0)
        
        file_type = uploaded_file.type
        
        if "pdf" in file_type.lower():
            # PDF 파일 처리
            img_bgr, total_pages = pdf_to_image(uploaded_file, page_num=page_num, dpi=300)
            return img_bgr, total_pages, "pdf"
        else:
            # 이미지 파일 처리
            uploaded_file.seek(0)
            data = uploaded_file.getvalue()
            
            if len(data) == 0:
                raise ValueError("파일이 비어있습니다")
            
            arr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("이미지 디코딩 실패")
            
            return img, 1, "image"
            
    except Exception as e:
        raise Exception(f"파일 로드 실패: {str(e)}")

def get_pdf_page_count(pdf_file):
    """PDF 페이지 수만 확인"""
    try:
        pdf_file.seek(0)
        pdf_data = pdf_file.read()
        pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")
        page_count = pdf_doc.page_count
        pdf_doc.close()
        pdf_file.seek(0)
        return page_count
    except:
        return 1

def align_images(A_bgr, B_bgr, nfeatures=4000):
    """
    A_bgr (기준)와 B_bgr를 특징점 매칭으로 정렬.
    반환: (A_bgr, warped_B_bgr, H, match_quality)
    """
    A_gray = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(A_gray, None)
    kp2, des2 = orb.detectAndCompute(B_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return A_bgr, None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return A_bgr, None, None, 0

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    min_matches = 10
    if len(good_matches) < min_matches:
        return A_bgr, None, None, len(good_matches) / min_matches

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
   
    if H is None:
        return A_bgr, None, None, 0

    inliers = np.sum(mask)
    match_quality = inliers / len(good_matches) if len(good_matches) > 0 else 0

    hA, wA = A_bgr.shape[:2]
    warped_B = cv2.warpPerspective(B_bgr, H, (wA, hA),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(255, 255, 255))

    return A_bgr, warped_B, H, match_quality

def fallback_align(A_bgr, B_bgr):
    """호모그래피 실패 시: B를 A 크기에 맞춰 리사이즈 후 중앙 배치"""
    hA, wA = A_bgr.shape[:2]
    hB, wB = B_bgr.shape[:2]

    scale = min(wA / wB, hA / hB)
    new_w = int(wB * scale)
    new_h = int(hB * scale)
    B_resized = cv2.resize(B_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((hA, wA, 3), 255, dtype=np.uint8)
    x_off = (wA - new_w) // 2
    y_off = (hA - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = B_resized
   
    return canvas

def compare_images(A_bgr, B_aligned_bgr, diff_thresh=30):
    """
    두 이미지 비교 (차이점 강조):
    - A에만 있는 부분 -> 파랑 (BGR: 255, 0, 0)
    - B에만 있는 부분 -> 빨강 (BGR: 0, 0, 255)
    - 공통 부분 -> 검정 (BGR: 0, 0, 0)
    - 배경 -> 흰색 (BGR: 255, 255, 255)
    """
    h, w = A_bgr.shape[:2]
   
    A_gray = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_aligned_bgr, cv2.COLOR_BGR2GRAY)
   
    _, A_bin = cv2.threshold(A_gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, B_bin = cv2.threshold(B_gray, 200, 255, cv2.THRESH_BINARY_INV)
   
    diff = cv2.absdiff(A_gray, B_gray)
    _, diff_mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
   
    only_A = np.logical_and(A_bin > 0, B_bin == 0).astype(np.uint8) * 255
    only_B = np.logical_and(B_bin > 0, A_bin == 0).astype(np.uint8) * 255
    both = np.logical_and(A_bin > 0, B_bin > 0).astype(np.uint8) * 255
   
    diff_common = np.logical_and(diff_mask > 0, both > 0)
    darker_in_A = np.logical_and(diff_common, A_gray < B_gray)
    darker_in_B = np.logical_and(diff_common, B_gray < A_gray)
   
    only_A = np.logical_or(only_A > 0, darker_in_A)
    only_B = np.logical_or(only_B > 0, darker_in_B)
    both = np.logical_and(both > 0, ~diff_mask.astype(bool))
   
    result = np.full((h, w, 3), 255, dtype=np.uint8)
    result[both] = [0, 0, 0]
    result[only_A] = [255, 0, 0]
    result[only_B] = [0, 0, 255]
   
    return result

def compare_images_overlay(A_bgr, B_aligned_bgr):
    """
    두 이미지를 겹쳐서 표시 (오버레이):
    - A (1번 도면) -> 주황색 (BGR: 0, 165, 255)
    - B (2번 도면) -> 초록색 (BGR: 0, 255, 0)
    """
    h, w = A_bgr.shape[:2]
   
    A_gray = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2GRAY)
    B_gray = cv2.cvtColor(B_aligned_bgr, cv2.COLOR_BGR2GRAY)
   
    _, A_bin = cv2.threshold(A_gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, B_bin = cv2.threshold(B_gray, 200, 255, cv2.THRESH_BINARY_INV)
   
    result = np.full((h, w, 3), 255, dtype=np.uint8)
   
    orange = np.array([0, 165, 255], dtype=np.uint8)
    result[A_bin > 0] = orange
   
    green = np.array([0, 255, 0], dtype=np.uint8)
    B_mask = B_bin > 0
    result[B_mask] = cv2.addWeighted(
        result[B_mask], 0.5,
        np.full_like(result[B_mask], green), 0.5,
        0
    )
   
    return result

def create_viewer_html(viewer_id_A, viewer_id_B, viewer_id_result, data_uris, layout="1:2"):
    """OpenSeadragon 뷰어 HTML 생성"""
    tile_sources_A = json.dumps({"type": "image", "url": data_uris[viewer_id_A]})
    tile_sources_B = json.dumps({"type": "image", "url": data_uris[viewer_id_B]})
    tile_sources_result = json.dumps({"type": "image", "url": data_uris[viewer_id_result]})
    
    if layout == "1:2":
        left_flex = "1"
        right_flex = "2"
    else:
        left_flex = "1"
        right_flex = "1"
   
    html = f"""
    <style>
        .container {{
            display: flex;
            gap: 10px;
            width: 100%;
            height: 800px;
        }}
        .left-panel {{
            flex: {left_flex};
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}
        .right-panel {{
            flex: {right_flex};
        }}
        .viewer-top {{
            flex: 1;
            border: 1px solid #ddd;
            min-height: 395px;
        }}
        .viewer-bottom {{
            flex: 1;
            border: 1px solid #ddd;
            min-height: 395px;
        }}
        .viewer-result {{
            width: 100%;
            height: 100%;
            border: 1px solid #ddd;
        }}
    </style>
   
    <div class="container">
        <div class="left-panel">
            <div id="{viewer_id_A}" class="viewer-top"></div>
            <div id="{viewer_id_B}" class="viewer-bottom"></div>
        </div>
        <div class="right-panel">
            <div id="{viewer_id_result}" class="viewer-result"></div>
        </div>
    </div>
   
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/3.0.0/openseadragon.min.js"></script>
    <script>
    var viewers = {{}};
    var syncing = false;
   
    viewers['{viewer_id_A}'] = OpenSeadragon({{
        id: "{viewer_id_A}",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/3.0.0/images/",
        tileSources: {tile_sources_A},
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        gestureSettingsMouse: {{
            scrollToZoom: true,
            clickToZoom: false,
            dblClickToZoom: true
        }},
        minZoomLevel: 0.5,
        maxZoomLevel: 10,
        zoomPerScroll: 1.2,
        animationTime: 0.3,
        timeout: 120000
    }});
   
    viewers['{viewer_id_B}'] = OpenSeadragon({{
        id: "{viewer_id_B}",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/3.0.0/images/",
        tileSources: {tile_sources_B},
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        gestureSettingsMouse: {{
            scrollToZoom: true,
            clickToZoom: false,
            dblClickToZoom: true
        }},
        minZoomLevel: 0.5,
        maxZoomLevel: 10,
        zoomPerScroll: 1.2,
        animationTime: 0.3,
        timeout: 120000
    }});
   
    viewers['{viewer_id_result}'] = OpenSeadragon({{
        id: "{viewer_id_result}",
        prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/3.0.0/images/",
        tileSources: {tile_sources_result},
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        gestureSettingsMouse: {{
            scrollToZoom: true,
            clickToZoom: false,
            dblClickToZoom: true
        }},
        minZoomLevel: 0.5,
        maxZoomLevel: 10,
        zoomPerScroll: 1.2,
        animationTime: 0.3,
        timeout: 120000
    }});
   
    function syncViewers(sourceViewer, sourceId) {{
        if (syncing) return;
        syncing = true;
       
        var center = sourceViewer.viewport.getCenter();
        var zoom = sourceViewer.viewport.getZoom();
       
        Object.keys(viewers).forEach(function(viewerId) {{
            if (viewerId !== sourceId) {{
                viewers[viewerId].viewport.panTo(center, null, false);
                viewers[viewerId].viewport.zoomTo(zoom, null, false);
            }}
        }});
       
        syncing = false;
    }}
   
    Object.keys(viewers).forEach(function(viewerId) {{
        var viewer = viewers[viewerId];
       
        viewer.addHandler('zoom', function(event) {{
            syncViewers(viewer, viewerId);
        }});
       
        viewer.addHandler('pan', function(event) {{
            syncViewers(viewer, viewerId);
        }});
       
        viewer.addHandler('open', function() {{
            console.log('Viewer ' + viewerId + ' loaded');
        }});
       
        viewer.addHandler('open-failed', function(event) {{
            console.error('Failed to load ' + viewerId);
            document.getElementById(viewerId).innerHTML =
                '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:red;">이미지 로드 실패</div>';
        }});
    }});
    </script>
    """
    return html

def process_and_display(file1, file2, diff_threshold, feature_count, mode="compare", 
                       page1=0, page2=0):
    """이미지 처리 및 표시"""
    try:
        # 파일 로드
        A_bgr, total_pages_A, type_A = load_file(file1, page_num=page1)
        B_bgr, total_pages_B, type_B = load_file(file2, page_num=page2)
       
        # PDF 정보 표시
        if type_A == "pdf" or type_B == "pdf":
            info_msg = []
            if type_A == "pdf":
                info_msg.append(f"1번 도면: PDF {total_pages_A}페이지 중 {page1+1}페이지")
            if type_B == "pdf":
                info_msg.append(f"2번 도면: PDF {total_pages_B}페이지 중 {page2+1}페이지")
            st.info(" | ".join(info_msg))
       
        # 정합 수행
        A_ref, warped_B, H, quality = align_images(A_bgr, B_bgr, nfeatures=feature_count)
       
        # 폴백 처리
        if warped_B is None or quality < 0.3:
            warped_B = fallback_align(A_ref, B_bgr)
       
        # 크기 일치 확인
        if warped_B.shape[:2] != A_ref.shape[:2]:
            warped_B = cv2.resize(warped_B, (A_ref.shape[1], A_ref.shape[0]))
       
        # 비교 수행
        if mode == "compare":
            result_bgr = compare_images(A_ref, warped_B, diff_thresh=diff_threshold)
        else:
            result_bgr = compare_images_overlay(A_ref, warped_B)
       
        # 레이아웃
        left_col, right_col = st.columns([1, 2])
       
        with left_col:
            st.markdown("### 업로드된 이미지")
            st.markdown("**1번 도면 (기준)**")
            st.markdown("")
            st.markdown("**2번 도면 (비교 대상)**")
       
        with right_col:
            if mode == "compare":
                st.markdown("### 비교 결과 (차이점 강조)")
                legend_col1, legend_col2, legend_col3 = st.columns(3)
                with legend_col1:
                    st.markdown("**🔵 파랑**: 1번만")
                with legend_col2:
                    st.markdown("**🔴 빨강**: 2번만")
                with legend_col3:
                    st.markdown("**⚫ 검정**: 공통")
            else:
                st.markdown("### 비교 결과 (오버레이)")
                legend_col1, legend_col2, legend_col3 = st.columns(3)
                with legend_col1:
                    st.markdown("**🟠 주황색**: 1번 도면")
                with legend_col2:
                    st.markdown("**🟢 초록색**: 2번 도면")
                with legend_col3:
                    st.markdown("**겹침**: 혼합 표시")
       
        # 뷰어 생성
        viewer_id_A = f"viewer_A_{random.randint(10000, 99999)}"
        viewer_id_B = f"viewer_B_{random.randint(10000, 99999)}"
        viewer_id_result = f"viewer_result_{random.randint(10000, 99999)}"
       
        all_images = {
            viewer_id_A: A_bgr,
            viewer_id_B: B_bgr,
            viewer_id_result: result_bgr
        }
       
        data_uris = {}
        for vid, img_bgr in all_images.items():
            h, w = img_bgr.shape[:2]
            max_dimension = 4000
           
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img_bgr
           
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                cv2.imwrite(tmp_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
           
            try:
                with open(tmp_path, "rb") as f:
                    data = f.read()
               
                file_size_mb = len(data) / (1024 * 1024)
                if file_size_mb > 5:
                    st.warning(f"이미지가 큽니다 ({file_size_mb:.1f}MB). 로딩이 느릴 수 있습니다.")
               
                data_b64 = base64.b64encode(data).decode("utf-8")
                data_uris[vid] = f"data:image/jpeg;base64,{data_b64}"
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
       
        html = create_viewer_html(viewer_id_A, viewer_id_B, viewer_id_result, data_uris, layout="1:2")
        st.components.v1.html(html, height=820)
       
        # 다운로드 버튼
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(result_rgb)
        buf = BytesIO()
        pil_result.save(buf, format="PNG")
        buf.seek(0)
       
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            filename = "drawing_comparison.png" if mode == "compare" else "drawing_overlay.png"
            st.download_button(
                "결과 다운로드 (PNG)",
                data=buf,
                file_name=filename,
                mime="image/png",
                use_container_width=True
            )
       
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        import traceback
        with st.expander("상세 오류 정보"):
            st.code(traceback.format_exc())

# -------------------------
# 탭 1: 도면 비교 (차이점 강조)
# -------------------------
with tab1:
    st.write("두 이미지 또는 PDF를 업로드하면 차이점을 색상으로 강조합니다. 모든 이미지는 마우스 스크롤로 줌/팬이 가능합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1_tab1 = st.file_uploader(
            "1번 도면 업로드", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"], 
            key="file1_tab1"
        )
    with col2:
        file2_tab1 = st.file_uploader(
            "2번 도면 업로드", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"], 
            key="file2_tab1"
        )
    
    # PDF 페이지 선택
    page1_tab1 = 0
    page2_tab1 = 0
    
    if file1_tab1 is not None or file2_tab1 is not None:
        col_page1, col_page2 = st.columns(2)
        
        if file1_tab1 is not None and "pdf" in file1_tab1.type:
            with col_page1:
                try:
                    total_pages_1 = get_pdf_page_count(file1_tab1)
                    if total_pages_1 > 1:
                        page1_tab1 = st.number_input(
                            f"1번 PDF 페이지 선택 (1-{total_pages_1})", 
                            min_value=1, 
                            max_value=total_pages_1, 
                            value=1,
                            key="page1_num_tab1"
                        ) - 1
                except:
                    pass
        
        if file2_tab1 is not None and "pdf" in file2_tab1.type:
            with col_page2:
                try:
                    total_pages_2 = get_pdf_page_count(file2_tab1)
                    if total_pages_2 > 1:
                        page2_tab1 = st.number_input(
                            f"2번 PDF 페이지 선택 (1-{total_pages_2})", 
                            min_value=1, 
                            max_value=total_pages_2, 
                            value=1,
                            key="page2_num_tab1"
                        ) - 1
                except:
                    pass
    
    with st.expander("고급 설정"):
        diff_threshold_tab1 = st.slider("차이 감지 임계값", 10, 100, 30, 
                                        help="낮을수록 작은 차이도 감지합니다", key="thresh_tab1")
        feature_count_tab1 = st.slider("특징점 개수", 1000, 10000, 4000, step=1000,
                                       help="많을수록 정확하지만 처리 시간이 길어집니다", key="feature_tab1")
    
    if file1_tab1 is not None and file2_tab1 is not None:
        process_and_display(file1_tab1, file2_tab1, diff_threshold_tab1, feature_count_tab1, 
                          mode="compare", page1=page1_tab1, page2=page2_tab1)
    else:
        st.info("좌우에 도면 이미지 또는 PDF를 모두 업로드해주세요.")
        with st.expander("사용 방법"):
            st.markdown("""
            1. **파일 업로드**: 비교할 두 도면(이미지 또는 PDF)을 업로드합니다.
            2. **PDF 페이지 선택**: PDF인 경우 비교할 페이지를 선택합니다.
            3. **자동 정합**: 프로그램이 자동으로 두 이미지를 정렬합니다.
            4. **차이 확인**: 결과 이미지에서 색상으로 차이를 확인합니다.
               - 파랑: 첫 번째 도면에만 있는 요소
               - 빨강: 두 번째 도면에만 있는 요소
               - 검정: 두 도면에 공통으로 있는 요소
            5. **줌/팬**: 마우스 스크롤로 확대/축소, 드래그로 이동 가능
            6. **다운로드**: 결과 이미지를 PNG로 저장 가능
            
            **지원 형식**: JPG, PNG, BMP, TIFF, PDF
            """)

# -------------------------
# 탭 2: 도면 겹치기 (오버레이)
# -------------------------
with tab2:
    st.write("두 이미지 또는 PDF를 겹쳐서 표시합니다. 모든 이미지는 마우스 스크롤로 줌/팬이 가능합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1_tab2 = st.file_uploader(
            "1번 도면 업로드", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"], 
            key="file1_tab2"
        )
    with col2:
        file2_tab2 = st.file_uploader(
            "2번 도면 업로드", 
            type=["jpg", "jpeg", "png", "bmp", "tiff", "pdf"], 
            key="file2_tab2"
        )
    
    # PDF 페이지 선택
    page1_tab2 = 0
    page2_tab2 = 0
    
    if file1_tab2 is not None or file2_tab2 is not None:
        col_page1, col_page2 = st.columns(2)
        
        if file1_tab2 is not None and "pdf" in file1_tab2.type:
            with col_page1:
                try:
                    total_pages_1 = get_pdf_page_count(file1_tab2)
                    if total_pages_1 > 1:
                        page1_tab2 = st.number_input(
                            f"1번 PDF 페이지 선택 (1-{total_pages_1})", 
                            min_value=1, 
                            max_value=total_pages_1, 
                            value=1,
                            key="page1_num_tab2"
                        ) - 1
                except:
                    pass
        
        if file2_tab2 is not None and "pdf" in file2_tab2.type:
            with col_page2:
                try:
                    total_pages_2 = get_pdf_page_count(file2_tab2)
                    if total_pages_2 > 1:
                        page2_tab2 = st.number_input(
                            f"2번 PDF 페이지 선택 (1-{total_pages_2})", 
                            min_value=1, 
                            max_value=total_pages_2, 
                            value=1,
                            key="page2_num_tab2"
                        ) - 1
                except:
                    pass
    
    with st.expander("고급 설정"):
        feature_count_tab2 = st.slider("특징점 개수", 1000, 10000, 4000, step=1000,
                                       help="많을수록 정확하지만 처리 시간이 길어집니다", key="feature_tab2")
    
    if file1_tab2 is not None and file2_tab2 is not None:
        process_and_display(file1_tab2, file2_tab2, 30, feature_count_tab2, 
                          mode="overlay", page1=page1_tab2, page2=page2_tab2)
    else:
        st.info("좌우에 도면 이미지 또는 PDF를 모두 업로드해주세요.")
        with st.expander("사용 방법"):
            st.markdown("""
            1. **파일 업로드**: 비교할 두 도면(이미지 또는 PDF)을 업로드합니다.
            2. **PDF 페이지 선택**: PDF인 경우 비교할 페이지를 선택합니다.
            3. **자동 정합**: 프로그램이 자동으로 두 이미지를 정렬합니다.
            4. **겹침 확인**: 결과 이미지에서 색상으로 확인합니다.
               - 주황색: 1번 도면의 선
               - 초록색: 2번 도면의 선
               - 겹치는 부분: 두 색이 혼합되어 표시
            5. **줌/팬**: 마우스 스크롤로 확대/축소, 드래그로 이동 가능
            6. **다운로드**: 결과 이미지를 PNG로 저장 가능
            
            **지원 형식**: JPG, PNG, BMP, TIFF, PDF
            """)
