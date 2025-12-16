import torch


class Layer(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Layer, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channels)

        self.conv2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channels)

        self.dwc = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn_skip = torch.nn.BatchNorm2d(output_channels)
        self.act = torch.nn.ReLU()

    def forward(self, x):
        x_skip = x.clone()
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + self.bn_skip(self.dwc(x_skip))
        return self.act(x)


class ConvBlockDense(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, act=torch.nn.ReLU(), bn=True):
        super(ConvBlockDense, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, kernel_size=3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = act
        self.use_bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class HookModule(torch.nn.Module):
    def __init__(self, module, height, width, input_channels=3, feature_layers=(), output_layers=(), feature_channels=None, output_channels=None):
        torch.nn.Module.__init__(self)
        self.module = module.cpu()

        if input_channels != 3:
            self.module.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.module.conv1.out_channels,
                                                kernel_size=self.module.conv1.kernel_size,
                                                padding=self.module.conv1.padding,
                                                bias=False)

        self.feature_layers = feature_layers
        self.output_layers = output_layers

        self.hooks = []
        self.features = []
        self.outputs = []
        self.register_hooks()

        self.feature_channels = []
        self.output_channels = []
        self.compute_channels_with_dummy(shape=(1, input_channels, height, width))

        self.feature_dconv = torch.nn.ModuleList()
        if feature_channels is not None:
            assert len(feature_channels) == len(self.feature_channels)
            self.feature_dconv = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
                    for cin, cout in zip(self.feature_channels, feature_channels)
                ]
            )
            self.feature_channels = feature_channels

        self.output_dconv = torch.nn.ModuleList()
        if output_channels is not None:
            assert len(output_channels) == len(self.output_channels)
            self.output_dconv = torch.nn.ModuleList(
                [
                    torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
                    for cin, cout in zip(self.output_channels, output_channels)
                ]
            )
            self.output_channels = output_channels

    def extract_layer(self, module, layer):
        if len(layer) == 0:
            return module
        else:
            return self.extract_layer(module._modules[layer[0]], layer[1:])

    def compute_channels_with_dummy(self, shape):
        dummy_input = torch.zeros(shape)
        self.module.forward(dummy_input)
        self.feature_channels = [f.shape[1] for f in self.features]
        self.output_channels = [o.shape[1] for o in self.outputs]
        self.features = []
        self.outputs = []

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def register_hooks(self):
        self.hooks = []
        self.features = []
        self.outputs = []
        features_hook = lambda m, i, o: self.features.append(o)
        outputs_hook = lambda m, i, o: self.outputs.append(o)
        for l in self.feature_layers:
            try:
                target = self.extract_layer(self.module, l.split("."))
                if hasattr(target, "_modules") and len(target._modules) > 0:
                    last = list(target._modules.values())[-1]
                    target = last
                hook_id = target.register_forward_hook(features_hook)
                self.hooks.append(hook_id)
            except Exception as e:
                print(f"[HookDebug][WARN] feature hook '{l}' failed: {repr(e)}")
        for l in self.output_layers:
            try:
                target = self.extract_layer(self.module, l.split("."))
                if hasattr(target, "_modules") and len(target._modules) > 0:
                    last = list(target._modules.values())[-1]
                    target = last
                hook_id = target.register_forward_hook(outputs_hook)
                self.hooks.append(hook_id)
            except Exception as e:
                print(f"[HookDebug][WARN] output hook '{l}' failed: {repr(e)}")

    def forward(self, x):
        self.remove_hooks()
        self.register_hooks()

        self.features = []
        self.outputs = []
        self.module(x)

        features = self.features
        if len(self.feature_dconv) > 0:
            features = [dconv(f) for f, dconv in zip(self.features, self.feature_dconv)]

        outputs = self.outputs
        if len(self.output_dconv) > 0:
            outputs = [dconv(o) for o, dconv in zip(self.outputs, self.output_dconv)]

        return features, outputs
    

    

# import torch


# class Layer(torch.nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(Layer, self).__init__()
#         self.conv1 = torch.nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
#         self.bn1 = torch.nn.BatchNorm2d(output_channels)

#         self.conv2 = torch.nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
#         self.bn2 = torch.nn.BatchNorm2d(output_channels)

#         self.dwc = torch.nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
#         self.bn_skip = torch.nn.BatchNorm2d(output_channels)
#         self.act = torch.nn.ReLU()

#     def forward(self, x):
#         x_skip = x.clone()
#         x = self.act(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         x = x + self.bn_skip(self.dwc(x_skip))
#         return self.act(x)


# class ConvBlockDense(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, bias=False, act=torch.nn.ReLU(), bn=True):
#         super(ConvBlockDense, self).__init__()
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=bias, kernel_size=3, stride=1, padding=1)
#         self.bn = torch.nn.BatchNorm2d(out_channels)
#         self.act = act
#         self.use_bn = bn

#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_bn:
#             x = self.bn(x)
#         if self.act is not None:
#             x = self.act(x)
#         return x


# class HookModule(torch.nn.Module):
#     """
#     Define the module, then you can determine which features are extracted, and which outputs are extracted.
#     For each you can decide if they are mapped to a lower dimension or not.

#     """
#     def __init__(self, module, height, width, input_channels=3, feature_layers=(), output_layers=(), feature_channels=None, output_channels=None):
#         torch.nn.Module.__init__(self)
#         self.module = module.cpu()

#         if input_channels != 3:
#             self.module.conv1 = torch.nn.Conv2d(in_channels=input_channels, out_channels=self.module.conv1.out_channels,
#                                                 kernel_size=self.module.conv1.kernel_size,
#                                                 padding=self.module.conv1.padding,
#                                                 bias=False)

#         self.feature_layers = feature_layers
#         self.output_layers = output_layers

#         self.hooks = []
#         self.features = []
#         self.outputs = []
#         self.register_hooks()

#         self.feature_channels = []
#         self.output_channels = []
#         self.compute_channels_with_dummy(shape=(1, input_channels, height, width))

#         self.feature_dconv = torch.nn.ModuleList()
#         if feature_channels is not None:
#             assert len(feature_channels) == len(self.feature_channels)
#             self.feature_dconv = torch.nn.ModuleList(
#                 [
#                     torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
#                     for cin, cout in zip(self.feature_channels, feature_channels)
#                 ]
#             )
#             self.feature_channels = feature_channels

#         self.output_dconv = torch.nn.ModuleList()
#         if output_channels is not None:
#             assert len(output_channels) == len(self.output_channels)
#             self.output_dconv = torch.nn.ModuleList(
#                 [
#                     torch.nn.Conv2d(in_channels=cin, out_channels=cout, kernel_size=1, stride=1, padding=0)
#                     for cin, cout in zip(self.output_channels, output_channels)
#                 ]
#             )
#             self.output_channels = output_channels

#     def extract_layer(self, module, layer):
#         if len(layer) == 0:
#             return module
#         else:
#             return self.extract_layer(module._modules[layer[0]], layer[1:])

#     def compute_channels_with_dummy(self, shape):
#         dummy_input = torch.zeros(shape)
#         self.module.forward(dummy_input)
#         self.feature_channels = [f.shape[1] for f in self.features]
#         self.output_channels = [o.shape[1] for o in self.outputs]
#         self.features = []
#         self.outputs = []

#     def remove_hooks(self):
#         for h in self.hooks:
#             h.remove()

#     def register_hooks(self):
#         self.features = []
#         self.outputs = []
#         features_hook = lambda m, i, o: self.features.append(o)
#         outputs_hook = lambda m, i, o: self.outputs.append(o)
#         # defensive: ensure module has the layers and attach hook to the last submodule if needed
#         for l in self.feature_layers:
#             try:
#                 target = self.extract_layer(self.module, l.split("."))
#                 # if target is Sequential, hook its last child
#                 if hasattr(target, "_modules") and len(target._modules) > 0:
#                     last = list(target._modules.values())[-1]
#                     target = last
#                 hook_id = target.register_forward_hook(features_hook)
#                 self.hooks.append(hook_id)
#             except Exception as e:
#                 print(f"[HookDebug][WARN] feature hook '{l}' failed: {repr(e)}")
#         for l in self.output_layers:
#             try:
#                 target = self.extract_layer(self.module, l.split("."))
#                 if hasattr(target, "_modules") and len(target._modules) > 0:
#                     last = list(target._modules.values())[-1]
#                     target = last
#                 hook_id = target.register_forward_hook(outputs_hook)
#                 self.hooks.append(hook_id)
#             except Exception as e:
#                 print(f"[HookDebug][WARN] output hook '{l}' failed: {repr(e)}")

#     def forward(self, x):
#         # debug: basic device/type info
#         # try:
#         #     print(f"[HookDebug] module_device={next(self.module.parameters()).device}, x_device={x.device}, x_shape={tuple(x.shape)}")
#         # except Exception as e:
#         #     print(f"[HookDebug] device info unavailable: {repr(e)}")

#         try:
#             self.remove_hooks()
#             self.register_hooks()
#         except Exception as e:
#             print(f"[HookDebug][WARN] re-register hooks failed: {repr(e)}")

#         self.features = []
#         self.outputs = []
#         self.module(x)

#         # print(f"[HookDebug] after forward: features={len(self.features)}, outputs={len(self.outputs)}")
#         features = self.features
#         if len(self.feature_dconv) > 0:
#             features = [dconv(f) for f, dconv in zip(self.features, self.feature_dconv)]

#         outputs = self.outputs
#         if len(self.output_dconv) > 0:
#             outputs = [dconv(o) for o, dconv in zip(self.outputs, self.output_dconv)]

#         if len(outputs) == 0:
#             print(f"[HookDebug][WARN] outputs is empty. feature_layers={self.feature_layers} output_layers={self.output_layers}")
#             # for i, f in enumerate(features):
#             #     try:
#             #         print(f"[HookDebug] features[{i}] shape={tuple(f.shape)}")
#             #     except Exception as e:
#             #         print(f"[HookDebug] features[{i}] shape unavailable: {repr(e)}")

#         return features, outputs