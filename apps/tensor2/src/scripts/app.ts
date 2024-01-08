import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

class App {
	private image: HTMLImageElement;
	private faceMatcher: faceapi.FaceMatcher;
	constructor() {
		window.addEventListener('DOMContentLoaded', async () => {
			this.image = document.querySelector('#jekim');
			await this.loadModels();
			await this.setFaceMatcher();
			await this.start();
		});
	}

	private async loadModels() {
		await Promise.all([
			faceapi.nets.tinyFaceDetector.loadFromUri('/models/face'),
			faceapi.nets.faceLandmark68Net.loadFromUri('/models/face'),
			faceapi.nets.ssdMobilenetv1.loadFromUri('/models/face'),
			faceapi.nets.faceRecognitionNet.loadFromUri('/models/face'),
		]);
	}

	private async setFaceMatcher() {
		const image = await faceapi.fetchImage('./assets/hannam.jpg');
		const detectionKim = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();

		const kimDescriper = new faceapi.LabeledFaceDescriptors('한남', [detectionKim.descriptor]);

		const faceMatcher = new faceapi.FaceMatcher(kimDescriper, 0.4);
		this.faceMatcher = faceMatcher;
	}

	private async start() {
		const canvasFrame = document.querySelector('#canvasFrame');
		const canvas = faceapi.createCanvasFromMedia(this.image);
		canvasFrame.append(canvas);
		const displaySize = { width: this.image.width, height: this.image.height };
		faceapi.matchDimensions(canvas, displaySize, true);

		const detections = await faceapi
			.detectAllFaces(this.image, new faceapi.SsdMobilenetv1Options())
			.withFaceLandmarks()
			.withFaceDescriptors();

		const results = await Promise.all(detections.map((d) => this.faceMatcher.findBestMatch(d.descriptor)));

		const resizedDetections = faceapi.resizeResults(detections, displaySize);

		results.forEach((result, i) => {
			const box = resizedDetections[i].detection.box;
			const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
			const ctx = canvas.getContext('2d');
			ctx.strokeStyle = 'white';
			ctx.setLineDash([5, 5]);
			drawBox.draw(canvas);
		});

		canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

		// faceapi.draw.drawDetections(canvas, resizedDetections);

		let hannamIndex = -1;
		let hannamScore = 0;
		for (let i = 0; i < results.length; i++) {
			const score = 1 - results[i].distance;
			if (score > hannamScore) {
				hannamIndex = i;
				hannamScore = score;
			}
		}

		for (let i = 0; i < resizedDetections.length; i++) {
			const detection = resizedDetections[i];
			if (i == hannamIndex) {
				const box = detection.detection.box;
				const faceCanvas = faceapi.createCanvasFromMedia(this.image);
				faceCanvas.width = box.width;
				faceCanvas.height = box.height;
				faceCanvas
					.getContext('2d')
					.drawImage(
						this.image,
						box.x,
						box.y,
						box.width,
						box.height,
						0,
						0,
						box.width,
						box.height
					);

				const imageData = faceCanvas.getContext('2d').getImageData(0, 0, box.width, box.height);
				const blurredImageData = await this.applyMosaicFilter(imageData, 10);
				canvas.getContext('2d').putImageData(blurredImageData, box.x, box.y);
			}
		}
	}

	private async applyMosaicFilter(imageData: ImageData, blockSize: number) {
		const width = imageData.width;
		const height = imageData.height;

		// 이미지 데이터의 픽셀 데이터 배열
		const pixels = imageData.data;

		// 가로, 세로 방향으로 모자이크 처리
		for (let y = 0; y < height; y += blockSize) {
			for (let x = 0; x < width; x += blockSize) {
				const startIndex = (y * width + x) * 4;

				// 현재 블록의 평균 색상 계산
				let totalRed = 0;
				let totalGreen = 0;
				let totalBlue = 0;

				// 블록 내의 픽셀들의 색상 값을 더한다
				for (let blockY = 0; blockY < blockSize; blockY++) {
					for (let blockX = 0; blockX < blockSize; blockX++) {
						const index = ((y + blockY) * width + x + blockX) * 4;
						totalRed += pixels[index];
						totalGreen += pixels[index + 1];
						totalBlue += pixels[index + 2];
					}
				}

				// 평균 색상 계산
				const avgRed = totalRed / (blockSize * blockSize);
				const avgGreen = totalGreen / (blockSize * blockSize);
				const avgBlue = totalBlue / (blockSize * blockSize);

				// 블록 내의 픽셀들을 평균 색상으로 설정
				for (let blockY = 0; blockY < blockSize; blockY++) {
					for (let blockX = 0; blockX < blockSize; blockX++) {
						const index = ((y + blockY) * width + x + blockX) * 4;
						pixels[index] = avgRed;
						pixels[index + 1] = avgGreen;
						pixels[index + 2] = avgBlue;
					}
				}
			}
		}

		return imageData;
	}
}

new App();
