import cv2
import os
import time

# Caminho para o classificador treinado
CLASSIFICADOR_PATH = "classificador_2/cascade.xml"

# Dimensão para redimensionar o frame da webcam
LARGURA_PADRAO = 640

# Pasta onde as imagens detectadas serão salvas
PASTA_SAIDA = "Img_cam"

# Inicializa a contagem de detecções
contador_deteccoes = 0

def preprocessar_imagem(imagem):
    h, w = imagem.shape[:2]
    escala = LARGURA_PADRAO / w
    nova_dim = (LARGURA_PADRAO, int(h * escala))
    imagem = cv2.resize(imagem, nova_dim)

    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    borrada = cv2.GaussianBlur(cinza, (5, 5), 0)
    equalizada = cv2.equalizeHist(borrada)
    return imagem, equalizada

def limpar_pasta_img_cam():
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
        return

    for nome_arquivo in os.listdir(PASTA_SAIDA):
        caminho_arquivo = os.path.join(PASTA_SAIDA, nome_arquivo)
        if os.path.isfile(caminho_arquivo):
            os.remove(caminho_arquivo)

def main():
    global contador_deteccoes

    print("🧹 Limpando a pasta de saída...")
    limpar_pasta_img_cam()

    print("✅ Carregando classificador...")
    classificador = cv2.CascadeClassifier(CLASSIFICADOR_PATH)
    if classificador.empty():
        print("❌ Falha ao carregar o classificador!")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Erro ao acessar a webcam.")
        return

    print("🎥 Webcam iniciada. Pressione 'q' para sair.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame.")
            break

        imagem_exibicao, imagem_pre = preprocessar_imagem(frame)

        deteccoes = classificador.detectMultiScale(
            imagem_pre,
            scaleFactor=1.25,
            minNeighbors=57,
            minSize=(40, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in deteccoes:
            contador_deteccoes += 1
            objeto = imagem_exibicao[y:y+h, x:x+w]
            nome_arquivo = f"det_{contador_deteccoes}_{int(time.time())}.jpg"
            caminho = os.path.join(PASTA_SAIDA, nome_arquivo)
            cv2.imwrite(caminho, objeto)
            cv2.rectangle(imagem_exibicao, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(imagem_exibicao, f"Detecções: {contador_deteccoes}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Detecção (pressione 'q' para sair)", imagem_exibicao)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"✅ Finalizado. Total de detecções: {contador_deteccoes}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
