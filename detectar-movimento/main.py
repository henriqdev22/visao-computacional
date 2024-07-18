import cv2

def inicializar_captura_video(caminho_video):
    return cv2.VideoCapture(caminho_video)

def preprocessar_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray

def processar_delta_frame(primeiro_frame, frame_cinza):
    delta_frame = cv2.absdiff(primeiro_frame, frame_cinza)
    limiar = cv2.threshold(delta_frame, 25, 255, cv2.THRESH_BINARY)[1]
    limiar = cv2.dilate(limiar, None, iterations=2)
    return delta_frame, limiar

def encontrar_e_desenhar_contornos(frame, limiar):
    contornos, _ = cv2.findContours(limiar.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contorno in contornos:
        if cv2.contourArea(contorno) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contorno)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def exibir_frames(frame, limiar, delta_frame):
    cv2.imshow("Principal", frame)
    cv2.imshow("Limiar", limiar)
    cv2.imshow("Delta do Frame", delta_frame)

def main():
    captura_video = inicializar_captura_video('detectar-movimento/caminhnado.mp4')  # Use 0 para webcam ou substitua pelo caminho do vídeo
    primeiro_frame = None
    while True:
        captura_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar no início do vídeo

        while True:
            ret, frame = captura_video.read()
            if not ret:
                break

            frame_cinza = preprocessar_frame(frame)

            if primeiro_frame is None:
                primeiro_frame = frame_cinza
                continue

            delta_frame, limiar = processar_delta_frame(primeiro_frame, frame_cinza)
            frame = encontrar_e_desenhar_contornos(frame, limiar)
            exibir_frames(frame, limiar, delta_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                captura_video.release()
                cv2.destroyAllWindows()
                return

    captura_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
