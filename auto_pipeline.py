import cv2
import os
import shutil
import subprocess

# Caminhos dos executáveis OpenCV
OPENCV_CREATESAMPLES = r"C:\opencv\build\x64\vc15\bin\opencv_createsamples.exe"
OPENCV_TRAINCASCADE = r"C:\opencv\build\x64\vc15\bin\opencv_traincascade.exe"

# Pastas
PASTA_POSITIVAS = "dataset/positivas"
PASTA_NEGATIVAS = "dataset/negativas"
PASTA_CLASSIFICADOR = "classificador"

# Arquivos
VEC_ARQUIVO = "amostras.vec"
ARQ_NEGATIVAS = "negativas.txt"
LISTA_AMOSTRAS = "amostras.lst"

# Parâmetros
WIN_W, WIN_H = 40, 80
NUM_STAGES = 10
NUM_NEGATIVAS = 100  # Ajuste conforme seu conjunto

def garantir_pastas():
    if os.path.exists(PASTA_CLASSIFICADOR):
        print("⚠️ Pasta classificador já existia. Removendo para novo treinamento...")
        shutil.rmtree(PASTA_CLASSIFICADOR)
    os.makedirs(PASTA_CLASSIFICADOR, exist_ok=True)

def gerar_negativas_txt():
    print("[Gerando negativas.txt]")
    with open(ARQ_NEGATIVAS, "w") as f:
        for nome in sorted(os.listdir(PASTA_NEGATIVAS)):
            if nome.lower().endswith((".jpg", ".jpeg", ".png")):
                caminho = os.path.join(PASTA_NEGATIVAS, nome)
                f.write(f"{caminho}\n")

def selecionar_objeto(imagem):
    bbox = cv2.selectROI("Selecione o objeto e pressione ENTER ou ESPAÇO", imagem, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecione o objeto e pressione ENTER ou ESPAÇO")
    x, y, w, h = bbox
    return (x, y, w, h) if w > 0 and h > 0 else None

def gerar_lista_amostras():
    print("[Selecionando objetos nas imagens positivas]")
    arquivos = sorted(os.listdir(PASTA_POSITIVAS))
    with open(LISTA_AMOSTRAS, "w") as f:
        for nome in arquivos:
            if not nome.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            caminho = os.path.join(PASTA_POSITIVAS, nome)
            imagem = cv2.imread(caminho)
            if imagem is None:
                continue

            selecao = selecionar_objeto(imagem)
            if selecao is None:
                print(f"⚠️ Pulando imagem {nome} (nenhuma região selecionada)")
                continue

            x, y, w, h = selecao
            f.write(f"{caminho} 1 {x} {y} {w} {h}\n")

def gerar_arquivo_vec():
    print("[Gerando arquivo .vec com opencv_createsamples]")
    with open(LISTA_AMOSTRAS, "r") as f:
        linhas = [l for l in f.readlines() if l.strip()]
    num_amostras = len(linhas)

    if num_amostras == 0:
        print("❌ Nenhuma amostra positiva encontrada.")
        return False

    comando = [
        OPENCV_CREATESAMPLES,
        "-info", LISTA_AMOSTRAS,
        "-num", str(num_amostras),
        "-w", str(WIN_W),
        "-h", str(WIN_H),
        "-vec", VEC_ARQUIVO
    ]

    resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if resultado.returncode != 0:
        print("❌ Erro ao gerar o .vec:")
        print(resultado.stderr)
        return False
    else:
        print("✅ .vec gerado com sucesso.")
        return True

def treinar_classificador():
    print("\n[Treinando classificador Haar Cascade]")

    with open(LISTA_AMOSTRAS, "r") as f:
        num_pos = len([l for l in f.readlines() if l.strip()]) - 5  # margem de segurança

    if num_pos <= 0:
        print("❌ Número de amostras positivas insuficiente.")
        return

    comando = [
        OPENCV_TRAINCASCADE,
        "-data", PASTA_CLASSIFICADOR,
        "-vec", VEC_ARQUIVO,
        "-bg", ARQ_NEGATIVAS,
        "-numPos", str(num_pos),
        "-numNeg", str(NUM_NEGATIVAS),
        "-numStages", str(NUM_STAGES),
        "-w", str(WIN_W),
        "-h", str(WIN_H)
    ]

    resultado = subprocess.run(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if resultado.returncode != 0:
        print("❌ Erro durante o treinamento:")
        print(resultado.stderr)
    else:
        print("✅ Treinamento finalizado. Classificador salvo em:", PASTA_CLASSIFICADOR)

def main():
    print("[Iniciando pipeline Haar Cascade simplificado]\n")
    garantir_pastas()
    gerar_negativas_txt()
    gerar_lista_amostras()
    if gerar_arquivo_vec():
        treinar_classificador()
        print("\n✅ Pipeline finalizado!")
    else:
        print("❌ Erro ao gerar .vec. Abortando.")

if __name__ == "__main__":
    main()
