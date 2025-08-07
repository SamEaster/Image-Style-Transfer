from PIL import Image
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms, models
from io import BytesIO
import requests

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

def load_image(img_path, max_size=512, shape=None):

    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if shape is not None:
        size = shape
    else:
        size = max_size if max(image.size) > max_size else max(image.size)
        
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                           (0.229, 0.224, 0.225))
    ])
    
    image = transform(image).unsqueeze(0).to(device)
    return image

def show_image(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze(0)
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)

    return image

def get_features(image, model, layers=None):
    if layers is None:
        layers = {
            '0': 'conv1_1',  
            '5': 'conv2_1',   
            '10': 'conv3_1',  
            '19': 'conv4_1',  
            '21': 'conv4_2',  
            '28': 'conv5_1'  
        }
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    batch, depth, height, width = tensor.size()
    tensor = tensor.view(depth, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram/ (height * width)

def show_image_color_corrected(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze(0)
    image = image.transpose(1,2,0)
    
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    image = image * std + mean
    
    image = np.clip(image, 0, 1)
    
    for i in range(3):
        channel = image[:, :, i]
        min_val, max_val = channel.min(), channel.max()
        if max_val > min_val: 
            image[:, :, i] = (channel - min_val) / (max_val - min_val)
    
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

def style_transfer_model(content_path, style_path, steps=50, content_weight=50, style_weight=3*1000):
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
    vgg.to(device).eval()
    
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    content = load_image(content_path, max_size=512)
    style = load_image(style_path, shape=content.shape[-2:])
    
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    target = content.clone().requires_grad_(True)
    
    style_weights = {
        'conv1_1': 1,
        'conv2_1':.8,
        'conv3_1':.6,
        'conv4_1':.4,
        'conv5_1':.2
    }
    
    optimizer = optim.Adam([target], lr=0.003)
    
    for step in range(1, steps + 1):
        target.data.clamp_(0, 1)
        
        optimizer.zero_grad()
        
        target_features = get_features(target, vgg)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            break
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_([target], max_norm=1.0)
        
        optimizer.step()
        if step % 1 == 0:
            print(f"Step [{step}/{steps}], Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")
    
    target.data.clamp_(0, 1)
    return target   