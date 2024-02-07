from torch.nn import Linear, Softmax
class ModelGS:
      def __init__(self, model, optimizer, criterion, data):
            self.optimizer = optimizer
            self.criterion = criterion
            self.model = model
            self.data = data
            self.m = Softmax(dim=1)

      def train(self):
            self.model.train()
            self.optimizer.zero_grad()
            # We now give as input also the graph connectivity
            out = self.model(self.data.x, self.data.edge_index)
            out = m(out)
            #out = model(data.x, data.edge_index,data.edge_weight)
            #print(len(out[data.train_mask]),len(data.y[data.train_mask]))
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            #loss = sigmoid_focal_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
            return loss

      def test(self,mask):
            self.model.eval()
            out = self.model(self.data.x, self.data.edge_index)
            #out = model(data.x, data.edge_index,data.edge_weight)
            pred = out.argmax(dim=1)
            test_correct = pred[mask] == self.data.y[mask]
            test_acc = int(test_correct.sum()) / int(mask.sum())
            test_out = out[mask]
            test_pred = pred[mask]
            return test_acc, test_out, test_pred
