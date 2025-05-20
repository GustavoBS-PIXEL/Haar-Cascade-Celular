import cv2
import os




# Caminho para o classificador treinado
CLASSIFICADOR_PATH = "classificador/cascade.xml"

# Caminho da imagem de teste
IMAGEM_TESTE_PATH = "img_teste/img_6.png"  # Altere para o caminho da imagem desejada

# Dimensão para redimensionar a imagem
LARGURA_PADRAO = 640





def preprocessar_imagem(imagem):
    # Redimensiona proporcionalmente
    h, w = imagem.shape[:2]
    escala = LARGURA_PADRAO / w
    nova_dim = (LARGURA_PADRAO, int(h * escala))
    imagem = cv2.resize(imagem, nova_dim)

    # Converte para escala de cinza
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplica blur para reduzir ruído
    borrada = cv2.GaussianBlur(cinza, (5, 5), 0)

    # Equaliza histograma para melhorar contraste
    equalizada = cv2.equalizeHist(borrada)

    return imagem, equalizada

def testar_classificador():
    print("✅ Carregando classificador...")
    classificador = cv2.CascadeClassifier(CLASSIFICADOR_PATH)

    if classificador.empty():
        print("❌ Falha ao carregar o classificador!")
        return
    else:
        print("✅ Classificador carregado com sucesso.")

    if not os.path.exists(CLASSIFICADOR_PATH):
        print("❌ Classificador não encontrado!")
        return

    classificador = cv2.CascadeClassifier(CLASSIFICADOR_PATH)
    imagem_original = cv2.imread(IMAGEM_TESTE_PATH)
    if imagem_original is None:
        print("❌ Erro ao carregar a imagem.")
        return

    imagem_exibicao, imagem_pre = preprocessar_imagem(imagem_original)

    deteccoes = classificador.detectMultiScale(
        imagem_pre,
        scaleFactor=1.015,
        minNeighbors=19,
        minSize=(40, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in deteccoes:
        cv2.rectangle(imagem_exibicao, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"📦 {len(deteccoes)} objetos detectados.")
    cv2.imshow("Resultado", imagem_exibicao)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    testar_classificador()
