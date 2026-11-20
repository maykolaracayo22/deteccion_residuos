# Detector de residuos con cámara

[Abrir la aplicación web](https://detectorresiduosvercel.vercel.app/)

Pulsa **Iniciar cámara**, permite el acceso y muestra un residuo. La aplicación dibuja una caja y muestra la clase, la confianza y el tacho sugerido. Puedes ajustar el umbral de confianza en la misma pantalla.

El detector YOLOv8n reconoce biodegradable, cartón, vidrio, metal, papel y plástico. ONNX Runtime Web procesa las imágenes en el navegador. Los pesos se incluyen en `public/models/waste_detector.onnx`.

## Ejecutar en la laptop

Con Node.js 22.12 o posterior, ejecuta desde la raíz del repositorio:

```sh
npm ci
npm run dev
```

Abre la dirección `localhost` que muestra Vite. Para usar la cámara desde un celular, abre el enlace HTTPS de la aplicación.

## Compilar

```sh
npm run build
```

La compilación genera `dist/`. El repositorio incluye la configuración de Vercel para publicar esa carpeta.
