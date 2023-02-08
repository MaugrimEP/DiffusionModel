from torchvision.transforms import transforms

from data.transforms.transforms_utils import ZeroNormalize

sample_transforms = transforms.Compose([
	transforms.Lambda(lambda img: img[:, :, :3]),  # remove the transparency component from png
	transforms.ToTensor(),
	ZeroNormalize(),
])

# initially used to sample a domain, not used currently
target_transforms = transforms.Compose([
	transforms.ToTensor(),
])
