import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from IPython.core.display import clear_output

class ConvSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=11, stride=1, dilation=1, p=0.0):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            # 1D Depth-wise
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=k,
                stride=stride,
                dilation=dilation,
                padding=dilation * (k - 1) // 2,
                bias=False
            ),
            # Pointwise (Time-wise)
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same"
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p)
        )
        

    def forward(self, x):
        return self.conv_block(x)

class MatchBoxBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=13, p=0.0):
        super().__init__()
        
        self.conv_block = ConvSubBlock(
            in_channels=in_channels, out_channels=out_channels, 
            k=k, p=p)     

    def forward(self, x):
        return self.conv_block(x)


def smooth(array, window_size):
    return [np.mean(array[max(0, i - window_size): i + 1]) for i in range(len(array))]

    
class MatchBox(LightningModule):
    def __init__(self, n_classes: int = 201):
        super().__init__()
        
        self.conv1 = ConvSubBlock(in_channels=128, out_channels=128, stride=2, k=11)
        
        self.box1 = MatchBoxBlock(in_channels=128, out_channels=64, k=13)
        self.box2 = MatchBoxBlock(in_channels=64, out_channels=64, k=15)
        self.box3 = MatchBoxBlock(in_channels=64, out_channels=64, k=17)
        
        self.conv2 = ConvSubBlock(in_channels=64, out_channels=128, dilation=2, k=29)
        self.conv3 = ConvSubBlock(in_channels=128, out_channels=128, k=1)
        
        self.end = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=n_classes,
                kernel_size=1,
            ),
            nn.AdaptiveMaxPool1d(output_size=1),
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # History
        self.train_loss_history = []
        self.val_loss_history = []
        
        self.train_acc_history = []
        self.train_fnr_history = []
        self.train_fpr_history = []
        self.val_acc_history = []
        self.val_fnr_history = []
        self.val_fpr_history = []
        
        self.last_train_outputs = None
        self.window_size = 5

        

    def forward(self, x) -> torch.Tensor:
        probs = self.end(self.conv3(self.conv2(
            self.box3(self.box2(self.box1(self.conv1(x))))
        )))
        probs = probs.squeeze(2)
        return probs

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        melspecs = batch['melspecs']
        labels = batch['labels']
        outputs = self.forward(melspecs)
        loss = self.criterion(outputs, labels)  
        
        pred_cl = outputs.argmax(dim=1)
        correct = torch.sum((pred_cl==labels))
        total = len(labels)
        
        fp = torch.sum((pred_cl!=labels)&(labels==0))
        neg = torch.sum(labels==0)
        fn = torch.sum((pred_cl!=labels)&(labels!=0))
        pos = torch.sum(labels!=0)
        
        self.log(f"train_loss", loss)
        return {"loss": loss, "correct": correct, "total": total,
               "fp": fp, "unknown": neg,
               "fn": fn, "phrases": pos}
    
    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        melspecs = batch['melspecs']
        labels = batch['labels']
        outputs = self.forward(melspecs)
        loss = self.criterion(outputs, labels)
        
        pred_cl = outputs.argmax(dim=1)
        correct = torch.sum((pred_cl==labels))
        total = len(labels)
        
        fp = torch.sum((pred_cl!=labels)&(labels==0))
        neg = torch.sum(labels==0)
        fn = torch.sum((pred_cl!=labels)&(labels!=0))
        pos = torch.sum(labels!=0)
        
        self.log(f"val_loss", loss)
        return {"loss": loss, "correct": correct, "total": total,
               "fp": fp, "unknown": neg,
               "fn": fn, "phrases": pos}
    
    
    def training_epoch_end(self, outputs):
        loss_values = [float(output['loss']) for output in outputs]
        avg_loss = np.mean(loss_values)
        
        total =  np.sum([output['total'] for output in outputs])
        acc = np.sum([output['correct'] for output in outputs]) / total       
                                   
        neg = np.sum([output['unknown'] for output in outputs])
        fpr = np.sum([output['fp'] for output in outputs]) / neg
            
        pos = np.sum([output['phrases'] for output in outputs])
        fnr = np.sum([output['fn'] for output in outputs]) / neg
        
        self.train_loss_history.append(avg_loss) 
        self.train_acc_history.append(acc)
        self.train_fnr_history.append(fnr)
        self.train_fpr_history.append(fpr)
          
        self.last_train_outputs = outputs
   
     
    def validation_epoch_end(self, outputs):
        avg_loss = np.mean([float(output['loss']) for output in outputs])
        self.val_loss_history.append(avg_loss)

        # train logs
        if self.last_train_outputs:
            clear_output()
            fig, axs = plt.subplots(1, 4, figsize=(18, 4))

            axs[0].plot(range(len(self.train_loss_history)), self.train_loss_history, label="loss")
            axs[0].plot(
                range(len(self.train_loss_history)),
                smooth(self.train_loss_history, window_size=self.window_size),
                label=f"{self.window_size}-smooth loss",
                c="aqua"
            )
            axs[0].set_xlabel("epoch")
            axs[0].legend()
            axs[0].grid(True)
            
            axs[1].plot(range(len(self.train_acc_history)), self.train_acc_history, label="accuracy")
            axs[1].set_xlabel("epoch")
            axs[1].set_ylim([0.0, 1.0])
            axs[1].legend()
            axs[1].grid(True)
        
            axs[2].plot(range(len(self.train_fpr_history)), self.train_fpr_history, label="fpr")
            axs[2].set_xlabel("epoch")
            axs[2].set_ylim([0.0, 1.0])
            axs[2].legend()
            axs[2].grid(True)
        
            axs[3].plot(range(len(self.train_fnr_history)), self.train_fnr_history, label="fnr")
            axs[3].set_xlabel("epoch")
            axs[3].set_ylim([0.0, 1.0])
            axs[3].legend()
            axs[3].grid(True)
            
            plt.show()

            
            total =  np.sum([output['total'] for output in self.last_train_outputs])
            acc = np.sum([output['correct'] for output in self.last_train_outputs]) / total
                                   
            neg = np.sum([output['unknown'] for output in self.last_train_outputs])
            fpr = np.sum([output['fp'] for output in self.last_train_outputs]) / neg
            
            pos = np.sum([output['phrases'] for output in self.last_train_outputs])
            fnr = np.sum([output['fn'] for output in self.last_train_outputs]) / pos
                               
            print(f"Train loss: {self.train_loss_history[-1]:.3f}\t" \
                  f"Accuracy: {acc}\t", f"Total: {total}\t")
            print(f"FPR: {fpr}\t", f"Total Unknown: {neg}\t")
            print(f"FNR: {fnr}\t", f"Total Act-Phrases: {pos}\t")

            
        # val metrics    
        total_v =  np.sum([output['total'] for output in outputs])
        acc = np.sum([output['correct'] for output in outputs]) / total_v       
                                   
        neg = np.sum([output['unknown'] for output in outputs])
        fpr = np.sum([output['fp'] for output in outputs]) / neg
            
        pos = np.sum([output['phrases'] for output in outputs])
        fnr = np.sum([output['fn'] for output in outputs]) / pos
        
        self.val_acc_history.append(acc)
        self.val_fnr_history.append(fnr)
        self.val_fpr_history.append(fpr)
            
        # val logs
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))

        axs[0].plot(range(len(self.val_loss_history)), self.val_loss_history, label="loss")
        axs[0].plot(
            range(len(self.val_loss_history)),
            smooth(self.val_loss_history, window_size=self.window_size),
            label=f"{self.window_size}-smooth loss",
            c="aqua"
        )
        axs[0].set_xlabel("epoch")
        axs[0].legend()
        axs[0].grid(True)
        
        axs[1].plot(range(len(self.val_acc_history)), self.val_acc_history, label="accuracy")
        axs[1].set_xlabel("epoch")
        axs[1].set_ylim([0.0, 1.0])
        axs[1].legend()
        axs[1].grid(True)
        
        axs[2].plot(range(len(self.val_fpr_history)), self.val_fpr_history, label="fpr")
        axs[2].set_xlabel("epoch")
        axs[2].set_ylim([0.0, 1.0])
        axs[2].legend()
        axs[2].grid(True)
        
        axs[3].plot(range(len(self.val_fnr_history)), self.val_fnr_history, label="fnr")
        axs[3].set_xlabel("epoch")
        axs[3].set_ylim([0.0, 1.0])
        axs[3].legend()
        axs[3].grid(True)
        
        plt.show()
        
        print(f"Val loss: {self.val_loss_history[-1]:.3f}\t" \
              f"Accuracy: {acc}\t", f"Total: {total_v}\t")
        print(f"FPR: {fpr}\t", f"Total Unknown: {neg}\t")
        print(f"FNR: {fnr}\t", f"Total Act-Phrases: {pos}\t")


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer     


if __name__ == '__main__':
    model = MatchBox(n_classes=201)
    print(model)
