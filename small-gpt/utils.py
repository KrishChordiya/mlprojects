# Define custom dataset
class QA_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        question = self.df.iloc[idx]['question']
        answer = self.df.iloc[idx]['answers']

        # input or prompt generally dont need eos token
        question_enc = self.tokenizer(
            question,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        answer_enc = self.tokenizer(
            answer+self.tokenizer.eos_token,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_id': question_enc['input_ids'].squeeze(),
            'input_mask': question_enc['attention_mask'].squeeze(),
            'label': answer_enc['input_ids'].squeeze(),
            'label_mask':answer_enc['attention_mask'].squeeze()
        }


def get_loader(dir, tokenizer, normalizer, batch_size, max_len=32):
    df = pd.read_csv(dir)

    # Normalize the questions and answers
    df['question'] = df['question'].apply(lambda x: normalizer.normalize_str(str(x)))
    df['answers'] = df['answers'].apply(lambda x: normalizer.normalize_str(str(x)))

    dataset = QA_Dataset(df, tokenizer, max_len)

    return DataLoader(dataset, batch_size=batch_size)


def train(model, train_loader, val_loader, loss_fn, optimizer, epochs, device):

    print(f'Training is done in {device}')

    model = model.to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for step, data in enumerate(train_loader):
            X = data['input_id'].to(device)
            y = data['label'].to(device)

            src_mask = data['input_mask'].unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = data['label_mask'].unsqueeze(1).unsqueeze(2) 
            
            future_mask = torch.tril(torch.ones(32, 32, dtype=torch.int)).unsqueeze(0).unsqueeze(0)# Upper triangle mask

            final_tgt_mask = (future_mask | tgt_mask).to(device) 
            output = model(X, y, src_mask, final_tgt_mask)
            loss = loss_fn(output.view(-1, output.shape[-1]), y.view(-1))
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.inference_mode():
            for data in val_loader:
                X = data['input_id'].to(device)
                y = data['label'].to(device)

                src_mask = data['input_mask'].unsqueeze(1).unsqueeze(2).to(device)
                tgt_mask = data['label_mask'].unsqueeze(1).unsqueeze(2) 
                
                future_mask = torch.tril(torch.ones(32, 32, dtype=torch.int)).unsqueeze(0).unsqueeze(0)# Upper triangle mask

                final_tgt_mask = (future_mask | tgt_mask).to(device) 
                output = model(X, y, src_mask, final_tgt_mask)
                loss = loss_fn(output.view(-1, output.shape[-1]), y.view(-1))
                val_loss += loss.item()

        val_loss /=len(val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss:{train_loss}, Validation Loss: {val_loss}')

      
            
    