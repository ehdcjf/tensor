import * as faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

class App {
	private video: HTMLVideoElement;
	playBtn: HTMLButtonElement;
	draw: boolean;
	constructor() {
		this.draw = false;
		window.addEventListener('DOMContentLoaded', () => {
			this.main();
		});
	}

	private async main() {
		await Promise.all([
			faceapi.nets.tinyFaceDetector.loadFromUri('/models/face'),
			faceapi.nets.faceLandmark68Net.loadFromUri('/models/face'),
			faceapi.nets.ssdMobilenetv1.loadFromUri('/models/face'),
			faceapi.nets.faceRecognitionNet.loadFromUri('/models/face'),
		]);
		this.video = document.querySelector('#video');

		// this.video.addEventListener('click', () => {
		// 	if (this.video.paused) this.video.play();
		// });
		this.video.addEventListener('play', () => {
			if (this.draw) return;
			else this.draw = true;
			const canvas = faceapi.createCanvasFromMedia(this.video);
			document.body.append(canvas);

			const displaySize = { width: this.video.width, height: this.video.height };

			faceapi.matchDimensions(canvas, displaySize);

			setInterval(async () => {
				const detections = await faceapi
					.detectAllFaces(this.video, new faceapi.SsdMobilenetv1Options())
					.withFaceLandmarks()
					.withFaceDescriptors();

				// const resizedDetections = faceapi.resizeResults(detections, displaySize);

				canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

				faceapi.draw.drawDetections(canvas, detections);

				for (const detection of detections) {
					const box = detection.detection.box;
					const faceCanvas = faceapi.createCanvasFromMedia(this.video);
					faceCanvas.width = box.width;
					faceCanvas.height = box.height;
					faceCanvas
						.getContext('2d')
						.drawImage(
							this.video,
							box.x,
							box.y,
							box.width,
							box.height,
							0,
							0,
							box.width,
							box.height
						);

					const imageData = faceCanvas
						.getContext('2d')
						.getImageData(0, 0, box.width, box.height);
					const blurredImageData = await this.applyMosaicFilter(imageData, 4);
					canvas.getContext('2d').putImageData(blurredImageData, box.x, box.y);
				}
			}, 10);
		});
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
