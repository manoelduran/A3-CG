import cv2
import numpy as np
from .segment_params import SegmentParams


def segment_single_bean(image: np.ndarray, params: SegmentParams) -> list[np.ndarray]:
    """
    Segmenta sementes com alto contraste de um fundo claro usando
    Limiarização de Otsu.

    Retorna uma lista de contornos válidos.
    """

    # 1. Converter para Tons de Cinza
    # Não precisamos de cor, apenas do contraste.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Segmentação (Limiarização)
    # Esta é a "mágica".
    # THRESH_BINARY_INV: Inverte (fundo fica preto, sementes brancas).
    # THRESH_OTSU: Encontra o valor de limiar ideal AUTOMATICAMENTE.
    # Adeus, _select_bean_clusters() e suas regras mágicas!
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 3. Limpeza (Opcional, mas recomendado)
    # Remove pequenos ruídos brancos na máscara.
    # Você já tinha isso (morphologyEx), é uma boa prática.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_limpa = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Encontrar Contornos (as "bordas" que você quer)
    contours, _ = cv2.findContours(
        mask_limpa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Filtrar contornos por área (como você já fazia)
    valid_contours = [
        c for c in contours if params.min_area <= cv2.contourArea(c) <= params.max_area
    ]

    print(f"Encontradas {len(valid_contours)} sementes.")

    # ------ Para Debug (Desenhar as caixas) ------
    debug_image = image.copy()
    for c in valid_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite("steps/0_resultado_simples.png", debug_image)
    cv2.imwrite("steps/1_mascara_otsu.png", mask_limpa)
    # ------ Fim do Debug ------

    return valid_contours
