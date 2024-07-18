import cv2
import pytesseract

def processa_imagem(img):
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Imagem Cinza', img_cinza)
    return img_cinza

def extrai_texto(img_cinza):
    try:
        texto = pytesseract.image_to_string(img_cinza, lang='por')
    except pytesseract.TesseractError as e:
        print(f"Erro ao usar o Tesseract: {e}")
        texto = ""
    return texto

def exibe_resultado(img, texto):
    print("Texto Extraído:")
    print(texto)
    
    cv2.imshow('Imagem Original', img)

    while True:
        key = cv2.waitKey(0)
        
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    image_path = 'print-para-texto/pablo.png'

    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return

    img_cinza = processa_imagem(img)
    texto = extrai_texto(img_cinza)
    exibe_resultado(img, texto)

if __name__ == "__main__":
    main()
