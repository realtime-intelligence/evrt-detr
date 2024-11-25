from torchvision import tv_tensors

def torchvision_isinstance(x, types):
    for cls in type(x).mro():
        if cls in types:
            return True
        elif cls is tv_tensors.TVTensor:
            break

    return False

