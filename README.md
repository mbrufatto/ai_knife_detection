# Detector de Objetos em Tempo Real com Alertas no Discord (YOLO & OpenCV)

Este projeto utiliza a biblioteca Ultralytics YOLO e OpenCV para detectar objetos específicos (configurado inicialmente para facas e tesouras) em tempo real a partir de uma webcam ou outra fonte de vídeo. Ele fornece uma interface gráfica simples (Tkinter) para selecionar dinamicamente o modelo YOLO e a fonte de vídeo, exibe a contagem de FPS e envia alertas para um canal do Discord configurado via webhook sempre que um objeto de alerta é detectado acima de um limiar de confiança, incluindo uma imagem do momento da detecção.

![Exemplo de uso](sample.png)

## Funcionalidades

- **Detecção de Objetos em Tempo Real:** Usa modelos YOLOv8 (ou outros compatíveis com Ultralytics) para detecção.
- **Interface Gráfica (GUI):** Construída com Tkinter para interação do usuário.
- **Seleção Dinâmica de Modelo:** Permite trocar o modelo YOLO `.pt` em tempo de execução através de um ComboBox.
- **Seleção Dinâmica de Fonte de Vídeo:** Permite alternar entre diferentes webcams ou fontes de vídeo (incluindo URLs RTSP) através de um ComboBox.
- **Exibição de FPS:** Mostra a taxa de quadros por segundo atual no canto da tela.
- **Alertas Configuráveis no Discord:**
  - Envia uma notificação para um webhook do Discord quando objetos específicos (`ALERT_CLASSES`) são detectados com confiança acima de um limiar (`ALERT_THRESHOLD`).
  - **Inclui Snapshot:** Anexa uma imagem `.png` do frame no momento da detecção ao alerta do Discord.
  - **Cooldown:** Possui um tempo de espera configurável (`ALERT_COOLDOWN_SECONDS`) entre alertas para evitar spam.
- **Operações Assíncronas:** Carregamento de modelos e envio de alertas para o Discord são feitos em threads separadas para não bloquear a interface do usuário.
- **Configurabilidade:** Várias opções (modelos, fontes, limiares, webhook) podem ser facilmente ajustadas no início do script Python.

## Setup Inicial (Usando Conda)

Siga estes passos para configurar o ambiente e executar o projeto localmente usando Conda.

**Pré-requisitos:**

- **Git:** Para clonar o repositório. ([Instalar Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- **Conda:** Gerenciador de pacotes e ambientes (Anaconda ou Miniconda). ([Instalar Miniconda](https://docs.conda.io/en/latest/miniconda.html))

**Passos:**

1.  **Clonar o Repositório:**

    ```bash
    git clone https://github.com/michaelycus/ai_knife_detection
    cd ai_knife_detection
    ```

2.  **Criar o Ambiente Conda:**
    Crie um novo ambiente virtual para isolar as dependências do projeto. Recomendamos Python 3.9 ou 3.10.

    ```bash
    conda create -n detector_env python=3.10
    ```

    Substitua `detector_env` pelo nome que preferir para o ambiente.

3.  **Ativar o Ambiente:**
    Antes de instalar qualquer coisa, ative o ambiente recém-criado:

    ```bash
    conda activate detector_env
    ```

    Você deverá ver `(detector_env)` no início do seu prompt de terminal.

4.  **Instalar PyTorch (IMPORTANTE):**
    Ultralytics YOLO depende do PyTorch. Instale-o _antes_ das outras dependências para garantir a versão correta (CPU ou GPU com suporte CUDA).

    - **Visite:** [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
    - Selecione suas opções (Stable, OS, Package: Conda ou Pip, Language: Python, Compute Platform: CPU ou sua versão CUDA).
    - **Execute o comando de instalação fornecido pelo site.** Exemplos comuns (use `pip` dentro do ambiente conda ativo):

      - **Para CPU:**
        ```bash
        pip install torch torchvision torchaudio
        ```
      - **Para GPU com CUDA 11.8 (Verifique sua versão!):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
      - **Para GPU com CUDA 12.1 (Verifique sua versão!):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```

5.  **Instalar as Dependências Restantes:**
    Com o PyTorch instalado e o ambiente `detector_env` ativo, instale as outras bibliotecas listadas no arquivo `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

6.  **Obter Modelos YOLO:**

    - Você precisará dos arquivos de modelo YOLO treinados (arquivos `.pt`).
    - Coloque os arquivos `.pt` (ex: `yolo11n.pt`, `yolo11s.pt`) na pasta raiz do projeto ou em um local acessível.
    - Certifique-se de que os caminhos no dicionário `AVAILABLE_MODELS` dentro do script Python (`main.py`) correspondem à localização dos seus arquivos de modelo.
    - Existem diversos modelos pré treinados do YOLO: [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt), [YOLO11s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt), [YOLO11m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt), [YOLO11l](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt) e [YOLO11x](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt). Para maIs detalhes, consultar a documentação do [YOLO](https://github.com/ultralytics/ultralytics).

7.  **Configurar o Script:**
    - Crie um arquivo `.env`.
    - **OBRIGATÓRIO:** Adicione a variável `DISCORD_WEBHOOK_URL` e edite a URL real do seu webhook do Discord.
      ```python
      DISCORD_WEBHOOK_URL = 'SUA_URL_DE_WEBHOOK_AQUI'
      ```
    - **Ajuste Opcional:** Modifique os dicionários `AVAILABLE_MODELS` e `AVAILABLE_SOURCES` para refletir seus modelos e câmeras disponíveis. Ajuste `DEFAULT_SOURCE_NAME`, `ALERT_CLASSES`, `ALERT_THRESHOLD` e `ALERT_COOLDOWN_SECONDS` conforme necessário.

## Como Usar

1.  **Ative o Ambiente Conda:**

    ```bash
    conda activate detector_env
    ```

2.  **Execute o Script:**
    Navegue até o diretório do projeto no terminal e execute:

    ```bash
    python main.py
    ```

3.  **Interaja com a GUI:**

    - A janela do aplicativo será aberta, exibindo o feed de vídeo da fonte padrão.
    - Use os ComboBoxes na parte superior para selecionar diferentes modelos YOLO ou fontes de vídeo.
    - O FPS será exibido no canto inferior direito.
    - Se um objeto de alerta for detectado acima do limiar configurado (e após o cooldown), um alerta com imagem será enviado para o Discord.

4.  **Sair do Programa:**
    Feche a janela da aplicação clicando no botão 'X'. O programa cuidará de liberar a câmera e encerrar corretamente.

## Dependências Principais

- **Python** (3.8+)
- **OpenCV (`opencv-python`)**: Para captura e processamento de vídeo/imagem.
- **PyTorch (`torch`)**: Backend de deep learning para YOLO.
- **Ultralytics (`ultralytics`)**: Para carregar e executar modelos YOLOv8.
- **Tkinter**: Para a interface gráfica (geralmente incluído no Python padrão).
- **Pillow (`PIL`)**: Para manipulação de imagens e integração com Tkinter.
- **Requests**: Para enviar os alertas HTTP para o Discord.

Veja o arquivo `requirements.txt` para a lista completa (exceto PyTorch, que deve ser instalado separadamente).

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais informações.
