import { Face } from './face';

class App {
	private main: HTMLElement;
	private srcFileInput: HTMLInputElement;
	private srcPreviewImage: HTMLImageElement;

	private findBtn: HTMLButtonElement;
	private face: Face;
	constructor() {
		this.face = new Face();
		this.main = document.querySelector('main');
		this.srcFileInput = document.querySelector('#srcFileInput');
		this.srcPreviewImage = document.querySelector('#srcPreviewImage');
		this.srcFileInput.addEventListener('change', (e) => this.showSrcPreview(e));

		this.findBtn = document.querySelector('#findBtn');
		this.findBtn.addEventListener('click', () => {
			if (this.srcFileInput.value) {
				this.face.start(this.srcPreviewImage);
			}
		});
	}

	showSrcPreview(e: Event) {
		const fileInput = e.target as HTMLInputElement;

		if (fileInput.files && fileInput.files[0]) {
			const reader = new FileReader();

			reader.onload = (e) => {
				this.srcPreviewImage.src = e.target.result as string;
				this.srcPreviewImage.width = 1200;
				this.srcPreviewImage.height = 800;
			};

			reader.readAsDataURL(fileInput.files[0]);
		}
	}
}

new App();
