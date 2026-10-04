# Hoja de ruta

1. **Probar en el Mac**: `clawde voice-test`, `clawde camera --index 0`, ajustar HSV y umbrales.
2. **Identificar la placa Adeept**: modelo, `platformio.ini`, compilar y subir el firmware de
   diagnóstico, `scripts/test_serial.py` (solo PING/STATUS).
3. **Medir el brazo**: pines, límites, home, velocidades, política de par y geometría
   (checklist en `calibration.md`).
4. **Servos en el firmware**: interpolación no bloqueante que STOP interrumpe, límites en la
   propia placa y watchdog.
5. **Cinemática** para la geometría medida y verificación a baja velocidad sin carga.
6. **Calibración mesa ↔ brazo** y primer pick & place supervisado, con el corte de
   alimentación a mano.
7. Ampliaciones: palabra de activación o escucha continua con VAD, detección de formas,
   corrección de distorsión, más colores o un modelo de visión local.
