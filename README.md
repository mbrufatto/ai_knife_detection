# 🛡️ FIAP VisionGuard: MVP para detecção supervisionada de objetos cortantes

## Descrição do Projeto

Este projeto utiliza a biblioteca **Ultralytics (8.3) YOLOv11** em conjunto com **OpenCV** para detectar objetos específicos em tempo real, como **facas** e **tesouras**, a partir de uma **webcam**, stream **RTSP** ou qualquer outra fonte de vídeo. 

A aplicação oferece uma interface gráfica simples (Tkinter) para selecionar dinamicamente o modelo YOLO (`.pt`) e a fonte de vídeo. Ela também exibe a taxa de FPS atual e envia alertas automáticos para um canal do Discord sempre que um objeto de interesse é detectado acima de um limiar de confiança, incluindo um snapshot do momento da detecção.

O modelo contém um treinamento preparado com datasets **RoboFlow** e um subconjunto filtrado do **dataset COCO 2017**. Este último contendo apenas classes relevantes como `knife`, `scissors` e `fork`.

As detecções e os alertas são processados de forma assíncrona, garantindo fluidez na interface e desempenho estável mesmo sem GPU.

## 🚀 Objetivos alcançados

### Detecção de Objetos em Tempo Real
- Baseada em **YOLOv11** (Ultralytics) para detectar objetos com precisão.
- Exibe a **taxa de FPS** atual diretamente na interface.

### Interface Gráfica (GUI)
- Desenvolvida com **Tkinter**, leve e nativa.
- **ComboBox** para:
  - Seleção de modelos `.pt` dinamicamente
  - Alternar entre **webcams locais** ou **streams RTSP**

### Integração com Discord (Alertas em Tempo Real)
- Envia alertas quando objetos específicos são detectados.
- **Snapshot do frame** incluído automaticamente no alerta.
- Comportamentos configuráveis:
    `ALERT_CLASSES` – classes que disparam alerta
    `ALERT_THRESHOLD` – nível de confiança mínimo
    `ALERT_COOLDOWN_SECONDS` – tempo de espera entre alertas

### Operações Assíncronas
- Modelos e alertas são processados em **threads paralelas**, mantendo a interface fluida.
- Interface **não bloqueia** durante detecção nem envio de alertas.

### Configurável e Extensível (início do script)
    ✅ Caminho para modelos  
    ✅ Lista de classes monitoradas  
    ✅ Webhook do Discord  
    ✅ Limiar de confiança  

### Exemplos de Aplicação
- Sistemas de monitoramento inteligente
- Ambientes com alto risco (escolas, hospitais, aeroportos)
- Detecção de objetos cortantes ou suspeitos  
![Exemplo de uso](images/sample.png)

## 📁 Estrutura de Pastas
    ├── data                                    - vídeos de avaliação
        └── inputs  
        └── results                             - Resultados obtidos com os modelos utilizados
            └── video_output_yolo11s_coco.mp4   - Saída do Vídeo de teste utilizando modelo yolo11s_dataset_coco.pt
            └── video2_output_yolo11s_coco.mp4  - Saída do Vídeo de teste utilizando modelo yolo11s_dataset_coco.pt
    ├── images                                  - imagens da documentação
    ├── models                                  - modelos treinados com datasets públicos
        └── yolo11m.pt                          - Modelo sem fine tuning (dataset Yolo), de tempo médio para uso com GPU 
        └── yolo11n_dataset_coco.pt             - Modelo com dataset Coco8, de treino com CPU
        └── yolo11n_dataset_knife_scissors.pt   - Modelo com dataset Roboflow, de rápido treino com CPU
        └── yolo11n.pt                          - Modelo sem fine tuning (dataset Yolo), de treino com CPU
        └── yolo11s_dataset_knife_scissors.pt   - Modelo com dataset Roboflow, com GPU média
        └── yolo11n_dataset_knife.pt            - Modelo com dataset Roboflow, especializado em facas para avalização
        └── yolo11s.pt                          - Modelo sem fine tuning (dataset Yolo), com GPU média
    ├── utils                                   - Scripts python para treinamento e testes rápidos dos modelos
        └── model_coach.py                      - Script para treino de modelos Yolo com dataset COCO
        └── model_tester.py                     - Script para teste rápido dos modelos treinados
    ├── main.py                                 - Projeto com captura de câmeras

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

## Resultados Obtidos


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

## Treinamento e utilização do modelo

Um novo modelo pode ser treinado a partir das instruções do arquivo `yolo_roboflow_training.ipynb`. Os modelos que foram utilizados para testar a aplicação podem ser encontrados [aqui](https://drive.google.com/drive/folders/1Fz7riVzD99UnTvr8JxoAzwQtpo5-nVh7?usp=sharing).

## Licença

Distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais informações.