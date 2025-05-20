Haar Cascade - Detecção de Celular
🎯 Objetivo
Treinar um classificador Haar Cascade personalizado para detectar um celular em imagens.

📚 Descrição do Projeto
Este projeto foi realizado em dupla para a disciplina de Visão Computacional.

Objeto detectado: Celular

Total de imagens positivas: 128

Total de imagens negativas: 242

🧪 Etapas Realizadas
Coleta de imagens positivas e negativas

Anotação manual das imagens positivas

Treinamento do classificador Haar Cascade com OpenCV

Testes com imagens novas para validação do modelo

⚙ Requisitos
Python 3.x

OpenCV instalado

OpenCV executáveis (opencv_createsamples.exe, opencv_traincascade.exe) da versão 3.4.11

📥 Instalação do OpenCV (executáveis)
Faça o download da versão correta do OpenCV com os utilitários necessários neste link:

👉 https://github.com/opencv/opencv/releases/download/3.4.11/opencv-3.4.11-vc14_vc15.exe

Após instalar:

Extraia os executáveis opencv_createsamples.exe e opencv_traincascade.exe localizados em build/x64/vc15/bin/

Aponte os caminhos corretos desses executáveis no script de treinamento (auto_pipeline.py)

🖥 Estrutura de Pastas

haar-cascade-celular/
│
├── dataset/
│   ├── positivas/
│   └── negativas/
│
├── classificador/
│   └── cascade.xml         # Será gerado após o treinamento
│
├── amostras.lst            # Lista de anotações positivas
├── negativas.txt           # Lista de imagens negativas
├── amostras.vec            # Arquivo de vetores para treinamento
├── detect_custom.py        # Script de teste/detecção
├── pipeline.py             # Script de treinamento
└── README.md

▶ Como Executar

1. Clonar o repositório
git clone https://github.com/seuusuario/haar-cascade-celular.git
cd haar-cascade-celular

3. Executar o pipeline de treinamento
Lembre-se de editar o caminho dos executáveis no início do pipeline.py.

3. Testar com uma imagem
Após o treinamento, o arquivo cascade.xml será gerado dentro da pasta classificador/.

📸 Resultados Esperados
O classificador irá desenhar caixas verdes ao redor de celulares detectados.
