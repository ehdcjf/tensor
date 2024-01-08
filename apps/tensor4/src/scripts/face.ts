import * as Faceapi from 'face-api.js';
import * as tf from '@tensorflow/tfjs';

export class Face {
	private faceMatcher: Faceapi.FaceMatcher;
	private canvas: HTMLCanvasElement;
	private hannamcon: HTMLImageElement;

	constructor() {
		this.init();
	}

	private async init() {
		await this.loadModels();
		await this.setFaceMatcher();
	}

	public async start(source: HTMLImageElement) {
		const canvasFrame = document.querySelector('#canvasFrame');
		const canvas = Faceapi.createCanvasFromMedia(source);
		canvasFrame.append(canvas);

		const displaySize = { width: source.width, height: source.height };
		Faceapi.matchDimensions(canvas, displaySize);
		const detections = await Faceapi.detectAllFaces(source, new Faceapi.SsdMobilenetv1Options())
			.withFaceLandmarks()
			.withFaceDescriptors();
		const resizedDetections = Faceapi.resizeResults(detections, displaySize);
		const results = await Promise.all(detections.map((d) => this.faceMatcher.findBestMatch(d.descriptor)));

		let hannamIndex = -1;
		let hannamScore = 0;
		for (let i = 0; i < results.length; i++) {
			const score = 1 - results[i].distance;
			if (score > hannamScore) {
				hannamIndex = i;
				hannamScore = score;
			}
		}

		canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
		Faceapi.draw.drawDetections(canvas, [resizedDetections[hannamIndex]]);

		const hannam = resizedDetections[hannamIndex];
		const box = hannam.detection.box;

		// const face = Faceapi.createCanvasFromMedia(source);
		// face.width = box.width;
		// face.height = box.height;

		const face = Faceapi.createCanvasFromMedia(this.hannamcon);
		face.width = box.width;
		face.height = box.height;

		const faceCtx = face.getContext('2d');

		// faceCtx.drawImage(source, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
		faceCtx.drawImage(
			this.hannamcon,
			0,
			0,
			this.hannamcon.width,
			this.hannamcon.height,
			0,
			0,
			box.width,
			box.height
		);
		const imageData = faceCtx.getImageData(0, 0, box.width, box.height);
		// const blurredImage = this.drawMosaic(imageData, 10);
		// canvas.getContext('2d').putImageData(blurredImage, box.x, box.y);
		canvas.getContext('2d').putImageData(imageData, box.x, box.y);
	}

	private async loadModels() {
		await Promise.all([
			Faceapi.nets.tinyFaceDetector.loadFromUri('/models/face'),
			Faceapi.nets.faceLandmark68Net.loadFromUri('/models/face'),
			Faceapi.nets.ssdMobilenetv1.loadFromUri('/models/face'),
			Faceapi.nets.faceRecognitionNet.loadFromUri('/models/face'),
		]);
	}

	private async setFaceMatcher() {
		const image = await Faceapi.fetchImage('./images/hannam.jpg');
		this.hannamcon = image;
		const detection = await Faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
		const descriptor = new Faceapi.LabeledFaceDescriptors('target', [detection.descriptor]);
		const faceMatcher = new Faceapi.FaceMatcher(descriptor, 0.4);
		this.faceMatcher = faceMatcher;
	}

	private drawMosaic(image: ImageData, blockSize: number) {
		const width = image.width;
		const height = image.height;

		// 이미지 데이터의 픽셀 데이터 배열
		const pixels = image.data;

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

		return image;
	}
}
