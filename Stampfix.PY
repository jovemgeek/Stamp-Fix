import tkinter as tk
from tkinter import ttk, filedialog, colorchooser, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk, ExifTags, ImageEnhance, ImageOps
import cv2
import numpy as np
import os
import time
import threading
import torch
global pipeline
from model.networks import Generator  # Certifique-se de importar o módulo correto aqui
keras_ocr_imported = False

# Função para importar o Keras OCR
def import_keras_ocr():
    global keras_ocr, keras_ocr_imported, pipeline
    import keras_ocr
    pipeline = keras_ocr.pipeline.Pipeline()
    keras_ocr_imported = True
    print("Keras OCR importado com sucesso!")

# Função para carregar o modelo Generator e configurar o dispositivo
def load_generator_model():
    import torchvision.transforms as T  # Importe torchvision.transforms aqui
    checkpoint_path = "pretrained/states_tf_celebahq.pth"
    # Configuração do dispositivo
    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')
    
    # Carregando o modelo
    generator_state_dict = torch.load(checkpoint_path)['G']
    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from model.networks import Generator
    else:
        from model.networks_tf import Generator  
    
    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)
    
    return generator, device, T


def show_splash(root):
    splash = tk.Toplevel(root)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    image = Image.open("splash.png")
    img_width, img_height = image.size

    x = (screen_width // 2) - (img_width // 2)
    y = (screen_height // 2) - (img_height // 2)
    splash.geometry(f"{img_width}x{img_height}+{x}+{y}")

    splash.title("Carregando...")
    splash.overrideredirect(True)
    splash.wm_attributes("-alpha", 0)  # tornar a janela completamente transparente

    image = image.convert("RGBA")
    data = image.getdata()
    new_data = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    image.putdata(new_data)

    tk_image = ImageTk.PhotoImage(image)
    splash_label = tk.Label(splash, image=tk_image, bg="white")
    splash_label.image = tk_image
    splash_label.pack(expand=True)

    splash.wm_attributes("-alpha", 1)  # tornar a janela opaca novamente

    while not keras_ocr_imported:  # Mantém o loop de fade in e fade out até a importação estar concluída
        # Efeito de fade in
        for i in range(0, 10):
            alpha = i / 10.0
            img = ImageEnhance.Brightness(image).enhance(alpha)
            tk_img = ImageTk.PhotoImage(img)
            splash_label.config(image=tk_img)
            splash_label.image = tk_img
            splash.update()
            time.sleep(0.1)

        # Efeito de fade out
        for i in range(9, -1, -1):
            alpha = i / 10.0
            img = ImageEnhance.Brightness(image).enhance(alpha)
            tk_img = ImageTk.PhotoImage(img)
            splash_label.config(image=tk_img)
            splash_label.image = tk_img
            splash.update()
            time.sleep(0.1)

    splash.destroy()


class TextDetectionApp:
    def __init__(self, root):
        global pipeline  # Use a variável global
        self.pipeline = pipeline  # Agora pipeline é uma variável de instância
        self.root = root
        self.image = None
        self.preview_image = None
        self.text_position = (0, 0)
        self.selection_rectangle = None
        self.start_x = None
        self.start_y = None
        self.image_paths = []  # Lista para armazenar os caminhos das imagens selecionadas
        self.image_texts = []  # Dicionário para armazenar o texto de cada imagem
        self.current_image_path = None  # Caminho da imagem atualmente selecionada
        self.images = []  # Lista para armazenar múltiplas imagens
        self.current_image_index = -1  # Index da imagem atualmente selecionada
        self.last_inserted_text = ""
        self.canvas_bind_id = None  # Adicionado para armazenar o ID do evento de ligação
        self.canvas_image = None
        self.original_canvas_width = None
        self.original_canvas_height = None
        # Obter a resolução da tela
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Calcular a posição x e y para colocar a janela no centro da tela
        x = (screen_width / 2) - (800 / 2)  # 800 é a largura da janela
        y = (screen_height / 2) - (800 / 2)  # 800 é a altura da janela
        
        root.geometry(f"800x700+{int(x)}+{int(y)}")  # Você pode ajustar a largura e a altura

        self.setup_gui(screen_width, screen_height)
        # Definir o tamanho mínimo da janela
        root.minsize(width=int(screen_width * 0.7), height=int(screen_height * 0.7))  # Ajuste esses valores

    def setup_gui(self, screen_width, screen_height):
        # Configure o estilo para cores e fundo escuro
        style = ttk.Style()
        style.configure("config_frame.TFrame", background="#555555")
        style.configure("TLabel", background="#555555", foreground="#ffffff")
        style.configure("TButton", background="#666666", foreground="#000000")

        self.root.title("StampFix")
        frame = ttk.Frame(self.root)
        frame.pack(padx=15, pady=15, expand=True, fill=tk.BOTH)

        # Ajustar o peso das linhas
        for i in range(12):
            frame.grid_rowconfigure(i, weight=1)
        frame.grid_rowconfigure(0, weight=2)  # Dê mais peso à linha onde o canvas está

        # Fazendo com que os widgets se expandam com a janela
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Seção de Configuração de Texto e Botões com fundo mais escuro
        config_frame = ttk.Frame(frame, style="config_frame.TFrame")
        config_frame.grid(row=0, column=0, rowspan=13, padx=5, pady=5, sticky="nsew")

        # Botão para selecionar imagem, agora com largura ajustada
        select_image_button = ttk.Button(config_frame, text="Selecionar Imagens", command=self.open_images, width=40)
        select_image_button.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        # Configurações de Texto
        labels = ["Tamanho da Fonte:", "Espessura da Borda:", "Cor da Borda:", "Cor do Texto:"]
        for i, label_text in enumerate(labels):
            label = ttk.Label(config_frame, text=label_text)
            label.grid(row=i + 2, column=0, padx=5, pady=5, sticky="w")

        self.font_size_var = tk.StringVar(value="22")
        font_size_entry = ttk.Entry(config_frame, textvariable=self.font_size_var)
        font_size_entry.grid(row=2, column=1, padx=5, pady=5)

        self.border_thickness_var = tk.StringVar(value="1")
        border_thickness_entry = ttk.Entry(config_frame, textvariable=self.border_thickness_var)
        border_thickness_entry.grid(row=3, column=1, padx=5, pady=5)

        self.border_color_var = tk.StringVar(value="#000000")
        border_color_button = ttk.Button(config_frame, text="Escolher cor", command=self.choose_border_color)
        border_color_button.grid(row=4, column=1, padx=5, pady=5)

        self.text_color_var = tk.StringVar(value="#FFFFFF")
        text_color_button = ttk.Button(config_frame, text="Escolher cor", command=self.choose_text_color)
        text_color_button.grid(row=5, column=1, padx=5, pady=5)

        # Campo de entrada de texto do usuário com cores atualizadas
        self.text_content = tk.Text(config_frame, height=5, width=30, bg="#666666", fg="#ffffff")
        self.text_content.grid(row=6, column=0, columnspan=3, padx=5, pady=5)
        self.text_content.bind("<KeyRelease>", self.multiple_actions)
        
        # Botão para remover texto
        remove_text_button = ttk.Button(config_frame, text="Remover Texto da Imagem selecionada", command=self.detect_and_remove_text, width=40)
        remove_text_button.grid(row=9, column=0, columnspan=3, padx=5, pady=5)

        # Botão para salvar imagem
        save_button = tk.Button(config_frame, text="Salvar Imagens", command=self.save_image, height=3, width=40)
        save_button.grid(row=14, column=0, columnspan=3, padx=5, pady=5)

        # Frame para os botões de seta
        arrow_frame = ttk.Frame(config_frame,style="config_frame.TFrame")
        arrow_frame.grid(row=8, column=0, columnspan=3, pady=2)

        # Adicionando botões de seta ao arrow_frame em forma de cruz
        up_arrow_button = ttk.Button(arrow_frame, text="↑", command=self.move_text_up)
        up_arrow_button.grid(row=0, column=1)

        left_arrow_button = ttk.Button(arrow_frame, text="←", command=self.move_text_left)
        left_arrow_button.grid(row=1, column=0)

        right_arrow_button = ttk.Button(arrow_frame, text="→", command=self.move_text_right)
        right_arrow_button.grid(row=1, column=2)

        down_arrow_button = ttk.Button(arrow_frame, text="↓", command=self.move_text_down)
        down_arrow_button.grid(row=2, column=1)
        
        remove_text_all_button = ttk.Button(config_frame, text="Remover Texto de Todas as Imagens", command=self.detect_and_remove_text_all)
        remove_text_all_button.grid(row=11, column=0, columnspan=3, padx=5, pady=5)

        # Adicionando Listbox para listar as imagens selecionadas dentro do config_frame
        self.image_listbox = tk.Listbox(config_frame, selectmode=tk.SINGLE, height=10, width=40)
        self.image_listbox.grid(row=12, column=0, columnspan=3, padx=5, pady=5)
        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        # Adicionando botão para remover a imagem selecionada da lista (ícone de lixeira)
        self.remove_image_button = tk.Button(config_frame, text="Remover Imagem / Limpar Lista", command=self.remove_selected_image, height=1, width=25)
        self.remove_image_button.grid(row=13, column=0, columnspan=3, padx=5, pady=5)
        # Ajuste para que os widgets usem todo o espaço vertical disponível
        for i in range(14):  # Supondo que você tenha 14 linhas no config_frame
            config_frame.grid_rowconfigure(i, weight=1)

        # Ajuste para que os widgets usem todo o espaço horizontal disponível
        config_frame.grid_columnconfigure(0, weight=1)
        # Canvas
        self.canvas = tk.Canvas(frame, cursor="cross")
        self.canvas.grid(row=0, column=1, rowspan=12, padx=5, pady=10, sticky="nsew")

        # Ajuste para permitir que o Canvas expanda
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Adiciona um evento de ligação para detectar o redimensionamento do Canvas
        self.canvas_bind_id = self.canvas.bind("<Configure>", self.on_canvas_resized)

        self.canvas_image = self.canvas.create_image(0, 0, anchor="nw")

        self.canvas.bind("<Button-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        # Definir o tamanho do Canvas e do frame de configuração com base na resolução da tela
        if screen_width >= 1920 and screen_height >= 1080:
            canvas_width = 900
            canvas_height = 800
            config_frame_height = 800
        elif screen_width >= 1366 and screen_height >= 768:
            canvas_width = 600
            canvas_height = 500
            config_frame_height = 500
        else:
            canvas_width = 400
            canvas_height = 350
            config_frame_height = 350

        # Definir o tamanho do Canvas e do frame de configuração
        self.canvas.config(width=canvas_width, height=canvas_height)
        config_frame.config(height=config_frame_height, width=200)  # Configurando a altura aqui
        # Sliders e botões de direção
        self.text_x_var = tk.DoubleVar(value=0.95)
        x_slider = ttk.Scale(frame, variable=self.text_x_var, orient=tk.HORIZONTAL, command=self.update_preview_dynamically)
        x_slider.grid(row=12, column=1, padx=5, pady=5, sticky="ew")
        # Sliders verticais e botões de direção movidos para o lado direito do canvas
        self.text_y_var = tk.DoubleVar(value=0.95)
        y_slider = ttk.Scale(frame, variable=self.text_y_var, orient=tk.VERTICAL, command=self.update_preview_dynamically)
        y_slider.grid(row=0, column=1, rowspan=12, padx=5, pady=5, sticky="nse")
        # Ajustando o layout
        for i in range(4):
            frame.columnconfigure(i, weight=1)
        for i in range(12):
            frame.rowconfigure(i, weight=1)

        self.save_button = save_button

    def update_preview_dynamically(self, *_):
        if not self.image:
            return

        self.update_canvas_with_text()
        
    def on_canvas_resized(self, event=None):
        if self.image:  # Somente se uma imagem foi carregada
            self.update_canvas()  # Redesenha a imagem para se ajustar ao novo tamanho
            self.scale_selection_rectangle()  # Novo método para escalar o retângulo de seleção

        # Novo método para escalar o retângulo de seleção

        
    # Novo método para escalar o retângulo de seleção
    def scale_selection_rectangle(self):
        if self.selection_rectangle is None:
            return
    
        if self.original_canvas_width is None or self.original_canvas_height is None:
            return
    
        # Calcule o fator de escala
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scale_x = canvas_width / self.original_canvas_width
        scale_y = canvas_height / self.original_canvas_height

        # Obtenha as coordenadas originais do retângulo de seleção
        try:
            coords = self.canvas.coords(self.selection_rectangle)
        except Exception as e:
            print(f"Erro ao tentar obter as coordenadas: {e}")
            return

        new_coords = [
            coords[0] * scale_x,
            coords[1] * scale_y,
            coords[2] * scale_x,
            coords[3] * scale_y
        ]

        # Atualize o retângulo de seleção com as novas coordenadas
        self.canvas.coords(self.selection_rectangle, new_coords)

        # Atualize as dimensões originais do canvas para o próximo redimensionamento
        self.original_canvas_width = canvas_width
        self.original_canvas_height = canvas_height


    def start_selection(self, event):
        if not self.image:
            return

        # Remover o retângulo anterior, se houver
        if self.selection_rectangle:
            self.canvas.delete(self.selection_rectangle)
    
        # Inicialize as dimensões originais do canvas se ainda não foram definidas
        if self.original_canvas_width is None:
            self.original_canvas_width = self.canvas.winfo_width()
        if self.original_canvas_height is None:
            self.original_canvas_height = self.canvas.winfo_height()

        # Calculando o deslocamento com base no tamanho atual do canvas e da imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        image_width = self.tk_image.width()
        image_height = self.tk_image.height()

        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        self.start_x = self.canvas.canvasx(event.x) - x_offset
        self.start_y = self.canvas.canvasy(event.y) - y_offset

        # Criando um retângulo de seleção
        self.selection_rectangle = self.canvas.create_rectangle(self.start_x + x_offset, self.start_y + y_offset,
                                                                self.start_x + x_offset, self.start_y + y_offset,
                                                                outline='red', width=2)

    def update_selection(self, event):
        if not self.selection_rectangle:
            return

        # Pega as dimensões do canvas e da imagem
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image_width = self.tk_image.width()
        image_height = self.tk_image.height()

        # Calcula os offsets
        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Calcula as coordenadas atuais com base no evento
        cur_x = self.canvas.canvasx(event.x) - x_offset
        cur_y = self.canvas.canvasy(event.y) - y_offset

        # Normaliza as coordenadas
        norm_cur_x = cur_x / image_width
        norm_cur_y = cur_y / image_height
        norm_start_x = self.start_x / image_width
        norm_start_y = self.start_y / image_height

        # Atualiza as coordenadas do retângulo de seleção
        self.canvas.coords(self.selection_rectangle, 
                           norm_start_x * image_width + x_offset, 
                           norm_start_y * image_height + y_offset, 
                           norm_cur_x * image_width + x_offset, 
                           norm_cur_y * image_height + y_offset)

    def move_text_left(self):
        current_value = self.text_x_var.get()
        if self.image:
            pixel_value = 0.001
            self.text_x_var.set(max(0, current_value - pixel_value))
            self.update_canvas_with_text()

    def move_text_right(self):
        current_value = self.text_x_var.get()
        if self.image:
            pixel_value = 0.001
            self.text_x_var.set(min(1, current_value + pixel_value))
            self.update_canvas_with_text()

    def move_text_up(self):
        current_value = self.text_y_var.get()
        if self.image:
            pixel_value = 0.001
            self.text_y_var.set(max(0, current_value - pixel_value))
            self.update_canvas_with_text()

    def move_text_down(self):
        current_value = self.text_y_var.get()
        if self.image:
            pixel_value = 0.001
            self.text_y_var.set(min(1, current_value + pixel_value))
            self.update_canvas_with_text()
    def multiple_actions(self, event):
        self.update_preview_dynamically()
        self.auto_save_text()

    def open_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        for file_path in file_paths:
            image = Image.open(file_path)

            # Corrigir a orientação da imagem com base nos metadados EXIF
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = image._getexif()
                if exif is not None and orientation in exif:
                    if exif[orientation] == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif exif[orientation] == 3:
                        image = image.rotate(180)
                    elif exif[orientation] == 4:
                        image = image.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
                    elif exif[orientation] == 5:
                        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif exif[orientation] == 6:
                        image = image.rotate(-90, expand=True)
                    elif exif[orientation] == 7:
                        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif exif[orientation] == 8:
                        image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # Casos em que os metadados EXIF estão ausentes ou corrompidos
                pass

            self.images.append(self.resize_image_to_fit(image, 800, 800))
            self.image_listbox.insert(tk.END, file_path.split('/')[-1])
            self.image_texts.append("")  # Inicialmente, sem texto

        if self.images:
            self.current_image_index = 0
            self.image = self.images[self.current_image_index]
            self.update_canvas()
            self.save_button.config(state=tk.NORMAL)

    def on_image_select(self, event):
        selected_index = self.image_listbox.curselection()
        if selected_index:
            self.current_image_index = selected_index[0]
            self.image = self.images[self.current_image_index]
        
            # Carregar o texto para a imagem atualmente selecionada
            current_text = self.image_texts[self.current_image_index]
            self.text_content.delete("1.0", tk.END)
        
            # Use o último texto inserido como sugestão se o campo estiver vazio
            if not current_text:
                current_text = self.last_inserted_text
        
            self.text_content.insert(tk.END, current_text)
        
            self.update_canvas_with_text()  # Atualize o canvas com o texto

    def update_text_for_current_image(self):
        if self.current_image_index != -1:
            current_text = self.text_content.get("1.0", "end-1c")
            self.image_texts[self.current_image_index] = current_text

    def remove_selected_image(self):
        print("Antes da remoção:", self.images, self.current_image_index)
        selected_index = self.image_listbox.curselection()
    
        if selected_index:
            self.image_listbox.delete(selected_index)
            del self.images[selected_index[0]]
            del self.image_texts[selected_index[0]]

            if self.images:
                self.current_image_index = 0 if selected_index[0] == 0 else selected_index[0] - 1
                self.image = self.images[self.current_image_index]
                self.update_canvas_with_text()
            else:
                self.current_image_index = -1  # Redefinir para -1 quando a lista estiver vazia
                self.image = None
                self.canvas.delete("all")
                # Se desejar, você pode desabilitar botões ou outras UI aqui
        else:
            confirm = messagebox.askyesno("Confirmação", "Nenhuma imagem selecionada. Deseja deletar todas as imagens?")
            if confirm:
                self.image_listbox.delete(0, tk.END)
                self.images.clear()
                self.image_texts.clear()
                self.current_image_index = -1
                self.image = None
                if self.canvas_image:
                    self.canvas.delete(self.canvas_image)
                    self.canvas_image = None

                # Se desejar, você pode desabilitar botões ou outras UI aqui
    
        print("Após a remoção:", self.images, self.current_image_index)
        self.on_image_select(None)  # Atualiza a seleção de imagem após a remoção

                
    def detect_and_remove_text(self):
        if not self.image:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        image_width = self.tk_image.width()
        image_height = self.tk_image.height()

        x_offset = (canvas_width - image_width) // 2
        y_offset = (canvas_height - image_height) // 2

        # Calculando os fatores de escala
        scale_x = self.image.width / image_width  # Use image_width que é a largura da imagem redimensionada para o canvas
        scale_y = self.image.height / image_height  # Use image_height que é a altura da imagem redimensionada para o canvas

        # Ajustando as coordenadas para a imagem original
        coords = self.canvas.coords(self.selection_rectangle)
        left = (min(coords[0], coords[2]) - x_offset) * scale_x
        upper = (min(coords[1], coords[3]) - y_offset) * scale_y
        right = (max(coords[0], coords[2]) - x_offset) * scale_x
        lower = (max(coords[1], coords[3]) - y_offset) * scale_y

        original_coords = (left, upper, right, lower)

        roi = self.image.crop(original_coords)
        roi.save("debug_roi.png")
        # Detectando texto na ROI
        cv_roi = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2BGR)
        predictions = self.pipeline.recognize([cv_roi])[0]

        mask = Image.new("L", self.image.size, 0)
        draw = ImageDraw.Draw(self.image)
        mask_draw = ImageDraw.Draw(mask)

            # Desenhar retângulos brancos na máscara nas posições onde o texto foi detectado
        for word, array in predictions:
            start_point = tuple(array[0])
            end_point = tuple(array[2])

            # Ajustar as coordenadas para a imagem original
            start_point = (int(start_point[0] + left), int(start_point[1] + upper))
            end_point = (int(end_point[0] + left), int(end_point[1] + upper))
            draw.rectangle([start_point, end_point], fill='white')
            mask_draw.rectangle([start_point, end_point], fill=255)
            
        self.image.save("debug_image_with_rect.png")
        mask.save("mask.png")
        self.apply_deepfill()
        # Atualizar a imagem na lista
        if self.current_image_index != -1:
            self.images[self.current_image_index] = self.image
        self.update_canvas()



    def apply_deepfill(self):

        temp_input_path = "temp_input.png"
        mask_path = "mask.png"
        output_path = "temp_output.png"

        
        self.image.save(temp_input_path)

        
        # Carregando imagem e máscara
        image = Image.open(temp_input_path)
        mask = Image.open(mask_path)
        
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)
        
        _, h, w = image.shape
        grid = 8
        
        image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)
        
        image = (image*2 - 1.).to(device)
        mask = (mask > 0.5).to(dtype=torch.float32, device=device)
        
        image_masked = image * (1.-mask)
        
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x*mask], dim=1)
        
        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)
            
        image_inpainted = image * (1.-mask) + x_stage2 * mask
        
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1)*127.5)
        img_out = img_out.to(device='cpu', dtype=torch.uint8)
        img_out = Image.fromarray(img_out.numpy())
        img_out.save(output_path)
        
        # Carregando a imagem resultante
        self.image = Image.open(output_path)


    def detect_and_remove_text_all(self):
        for idx, image in enumerate(self.images):
            self.current_image_index = idx
            self.image = image
            self.detect_and_remove_text()
            self.images[idx] = self.image
        self.update_canvas_with_text()

    def resize_image_to_fit(self, image, width, height):
        # Obter os dados EXIF da imagem original
        exif_data = image.info.get('exif')
    
        # Lógica atual de redimensionamento
        img_width, img_height = image.size
        scale_factor = min(width / img_width, height / img_height)
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
        # Anexar os dados EXIF à imagem redimensionada
        if exif_data:
            resized_image.info['exif'] = exif_data

        return resized_image



    def update_canvas(self):
        if not self.image:
            return

        if self.canvas_bind_id:
            self.canvas.unbind("<Configure>", self.canvas_bind_id)  # Unbind to avoid infinite loop

        # Resize image based on Canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        resized_image = self.resize_image_to_fit(self.image, canvas_width, canvas_height)

        # Add text to the image if there is any
        draw = ImageDraw.Draw(resized_image)
        font_size = int(self.font_size_var.get())
        font = ImageFont.truetype("arial.ttf", font_size)

        text_content = self.text_content.get("1.0", "end-1c")
        border_thickness = int(self.border_thickness_var.get())
        border_color = self.border_color_var.get()
        text_color = self.text_color_var.get()

        img_width, img_height = resized_image.size
        text_width, text_height = draw.textbbox((0, 0), text_content, font=font)[2:4]
        x_position = img_width - text_width - 5  # 5 pixels of margin
        y_position = img_height - text_height - 5  # 5 pixels of margin

        # Updating text position based on sliders
        x_position += (self.text_x_var.get() - 0.95) * img_width  # adjusted for the initial value of the slider
        y_position += (self.text_y_var.get() - 0.95) * img_height  # adjusted for the initial value of the slider

        text_position = (x_position, y_position)
        draw.multiline_text(text_position, text_content, fill=text_color, font=font, align="right", stroke_width=border_thickness, stroke_fill=border_color)
    
        # Create Tk image
        self.tk_image = ImageTk.PhotoImage(resized_image)

        # Center the image on the canvas
        x = (canvas_width - self.tk_image.width()) // 2
        y = (canvas_height - self.tk_image.height()) // 2

        # If canvas image doesn't exist, create it. Else, update it.
        if self.canvas_image is None:
            self.canvas_image = self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        else:
            self.canvas.itemconfig(self.canvas_image, image=self.tk_image)
            self.canvas.coords(self.canvas_image, x, y)

        # Re-bind the event
        self.canvas_bind_id = self.canvas.bind("<Configure>", self.on_canvas_resized)

    def save_image(self):
        # Atualize o texto da imagem atualmente selecionada
        self.update_text_for_current_image()

        if not self.images:  # Verifique se a lista de imagens está vazia
            return

        # Peça ao usuário para selecionar a pasta onde as imagens serão salvas
        folder_selected = filedialog.askdirectory()
        if not folder_selected:  # Se o usuário cancelar a seleção
            return

        # Função interna para adicionar texto à imagem
        def add_text_to_image(image, text_content):
            draw = ImageDraw.Draw(image)
            font_size = int(self.font_size_var.get())
            font = ImageFont.truetype("arial.ttf", font_size)
            border_thickness = int(self.border_thickness_var.get())
            border_color = self.border_color_var.get()
            text_color = self.text_color_var.get()

            img_width, img_height = image.size
            text_width, text_height = draw.textbbox((0, 0), text_content, font=font)[2:4]
            x_position = img_width - text_width - 5  # 5 pixels de margem
            y_position = img_height - text_height - 5  # 5 pixels de margem

            x_position += (self.text_x_var.get() - 0.95) * img_width
            y_position += (self.text_y_var.get() - 0.95) * img_height

            text_position = (x_position, y_position)

            draw.multiline_text(text_position, text_content, fill=text_color, font=font, align="right", stroke_width=border_thickness, stroke_fill=border_color)

        # Função interna para corrigir a orientação da imagem
        def fix_image_orientation(image):
            try:
                for orientation in ImageOps.ExifTags.TAGS.keys():
                    if ImageOps.ExifTags.TAGS[orientation] == 'Orientation':
                        break

                exif = image._getexif()
                if exif:
                    exif = dict(exif.items())
                    if orientation in exif:
                        if exif[orientation] == 3:
                            image = image.rotate(180, expand=True)
                        elif exif[orientation] == 6:
                            image = image.rotate(270, expand=True)
                        elif exif[orientation] == 8:
                            image = image.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError):
                # Se não for possível corrigir a orientação, retorne a imagem original
                pass

            return image

        if len(self.images) > 1:
            for idx, image in enumerate(self.images):
                image_to_save = fix_image_orientation(image.copy())
                text_content = self.image_texts[idx]
                add_text_to_image(image_to_save, text_content)
                original_filename = self.image_listbox.get(idx)
                save_path = os.path.join(folder_selected, original_filename)
                image_to_save.save(save_path)
        else:
            image_to_save = fix_image_orientation(self.image.copy())
            text_content = self.image_texts[0] if self.image_texts else ""
            add_text_to_image(image_to_save, text_content)
            file_path = filedialog.asksaveasfilename(initialdir=folder_selected, defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                image_to_save.save(file_path)
            
        # Funções adicionais para adicionar texto
    def choose_text_color(self):
        color = colorchooser.askcolor(title="Escolher cor do texto")[1]
        if color:
            self.text_color_var.set(color)
            self.update_canvas_with_text()

    def choose_border_color(self):
        color = colorchooser.askcolor(title="Escolher cor da borda")[1]
        if color:
            self.border_color_var.set(color)
            self.update_canvas_with_text()
            
    def auto_save_text(self, event=None):
        if self.current_image_index != -1:  # Se uma imagem está selecionada
            current_text = self.text_content.get("1.0", "end-1c")
            self.image_texts[self.current_image_index] = current_text
            self.last_inserted_text = current_text  # Atualiza o último texto inserido

    def update_canvas_with_text(self):
        self.update_canvas()



if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal enquanto o splash screen é exibido

    # Crie e inicie um novo thread para importar o Keras OCR
    keras_thread = threading.Thread(target=import_keras_ocr)
    keras_thread.start()

    # Chame a função para carregar o modelo Generator e configurar o dispositivo
    generator, device, T = load_generator_model()

    # Mostrar o splash
    show_splash(root)

    # Iniciar o aplicativo
    app = TextDetectionApp(root)

    # Aguarde até que o thread do Keras OCR termine (opcional, dependendo de onde você precisa usar o Keras OCR)
    keras_thread.join()

    # Mostra a janela principal após o splash screen
    root.deiconify()

    # Iniciar o loop principal
    root.mainloop()
