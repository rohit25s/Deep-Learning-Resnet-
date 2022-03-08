# Code Structure
Pytorch version=1.7.0
All code files are within code folder <br/>
Saved_models has saved models which can be used as checkpoints <br/>
predictions.npy has predictions for private dataset <br/>

# To run project
### To train the model  
```
python main.py
```

### To test on private dataset
```
# load saved model
model = MyModel()
model.network.load_state_dict(torch.load(checkpoint_path))

model.evaluate(test_data)

# load private dataset into test_loader
private_loader = load_testing_images(path_to_private_dataset, preprocess_config)

#predict probability
model.pred_prob(private_loader)
```


	