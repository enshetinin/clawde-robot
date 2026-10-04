# Calibración

## Mesa (cámara → mm)

Homografía del plano de la mesa a partir de al menos 4 correspondencias píxel ↔ mm, sin 3
puntos alineados ni duplicados. Se usa un DLT normalizado y se rechaza si el error RMS supera
`max_calibration_rms_mm`.

1. Fija la cámara y marca 4 o más puntos en la mesa con posiciones medidas en mm (por ejemplo,
   las esquinas de una hoja A3, con el origen en la base del brazo).
2. Lee sus píxeles con `clawde camera --index 0 --save` (guarda una captura) y un visor de
   imágenes.
3. Crea `puntos.json`:

```json
{
  "image_size": [1280, 720],
  "camera_index": 0,
  "camera_name": "Logitech",
  "points": [
    {"px": [312, 205], "mm": [-150, 300]},
    {"px": [968, 210], "mm": [150, 300]},
    {"px": [1010, 640], "mm": [150, 90]},
    {"px": [270, 635], "mm": [-150, 90]}
  ]
}
```

4. `clawde calibrate --points puntos.json` guarda `config/table_calibration.json` con la
   resolución, la cámara, la fecha y el error.

Supuestos: la mesa es plana, la distorsión de la lente es despreciable o está corregida, y la
cámara no se mueve. Con otra resolución la calibración se rechaza: hay que recalibrar o
validarla explícitamente. **Z y la forma de agarre no salen de la homografía.** Sin
calibración, CLAWDE muestra píxeles y permite simulación, pero el modo real queda bloqueado.

## Brazo (pendiente)

Antes de pensar en `--real` hacen falta, medidos y verificados:

- [ ] Modelo exacto de la placa Adeept y su `platformio.ini`
- [ ] Puerto serie (`clawde devices serial`; ahora aparece `/dev/cu.usbserial-140`, sin verificar)
- [ ] Por servo: pin, mínimo y máximo reales, home y velocidad máxima
- [ ] Si los servos mantienen par tras STOP (si no, el brazo podría caer)
- [ ] Geometría: tipo y longitudes de los eslabones
- [ ] Espacio de trabajo (x/y/z en mm)
- [ ] Solucionador IK para esa geometría, verificado físicamente a baja velocidad
- [ ] Transformación mesa ↔ base del brazo
- [ ] Después: `calibrated: true`, `hardware_enabled: true` y `driver: serial` en `robot.yaml`
