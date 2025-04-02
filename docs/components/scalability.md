# Scalability Infrastructure

This document outlines our approach to scaling the Neural Circuit Extraction and Modular Composition Framework across multiple dimensions: model size, dataset complexity, training throughput, and circuit analysis.

## Core Scalability Principles

Our architecture is built on these foundational scalability principles:

1. **Horizontal Scalability**: Add more compute resources to linearly increase throughput
2. **Hierarchical Decomposition**: Break large problems into independently solvable subproblems
3. **Dynamic Resource Allocation**: Allocate compute resources based on task priority and complexity
4. **Incremental Processing**: Process and store data incrementally rather than requiring full dataset loads
5. **Smart Caching**: Cache intermediate results strategically to prevent redundant computation

## Distributed Training Infrastructure

Our training infrastructure scales across multiple nodes:

### Distributed Data Parallelism

```python
class DistributedTrainer:
    def __init__(self, model, optimizer_fn, world_size, rank):
        self.local_model = model
        self.world_size = world_size
        self.rank = rank
        
        # Wrap model for distributed training
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.local_model,
            device_ids=[rank] if torch.cuda.is_available() else None
        )
        
        # Initialize optimizer
        self.optimizer = optimizer_fn(self.model.parameters())
        
    def train_epoch(self, dataloader, loss_fn):
        """Train for one epoch with distributed data parallelism."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move data to appropriate device
            inputs = inputs.to(self.rank)
            targets = targets.to(self.rank)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()
            
        # Average loss across all batches
        return epoch_loss / len(dataloader)
```

### Model Parallelism

For very large models that don't fit on a single device:

```python
class ModelParallelTransformer:
    def __init__(self, config, num_devices):
        self.config = config
        self.num_devices = num_devices
        
        # Split layers across devices
        self.devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
        layers_per_device = config.num_layers // num_devices
        
        # Create layer groups
        self.layer_groups = []
        for i in range(num_devices):
            start_layer = i * layers_per_device
            end_layer = (i + 1) * layers_per_device if i < num_devices - 1 else config.num_layers
            
            # Create layers for this device
            device_layers = nn.ModuleList([
                TransformerLayer(config) 
                for _ in range(start_layer, end_layer)
            ]).to(self.devices[i])
            
            self.layer_groups.append(device_layers)
        
        # Embedding on first device, output on last device
        self.embedding = TokenEmbedding(config).to(self.devices[0])
        self.output_layer = OutputLayer(config).to(self.devices[-1])
        
    def forward(self, input_ids):
        # Start on first device
        x = self.embedding(input_ids.to(self.devices[0]))
        
        # Process through layer groups
        for device_idx, layers in enumerate(self.layer_groups):
            for layer in layers:
                x = layer(x)
                
            # Transfer to next device if not the last one
            if device_idx < self.num_devices - 1:
                next_device = self.devices[device_idx + 1]
                x = x.to(next_device)
        
        # Final output projection
        return self.output_layer(x)
```

### Pipeline Parallelism

For efficient processing of very deep models:

```python
class PipelineParallelTrainer:
    def __init__(self, partitioned_model, optimizer_fns, num_microbatches=32):
        self.partitioned_model = partitioned_model  # List of model partitions
        self.num_stages = len(partitioned_model)
        self.num_microbatches = num_microbatches
        
        # Create optimizers for each partition
        self.optimizers = [
            opt_fn(model.parameters()) 
            for model, opt_fn in zip(partitioned_model, optimizer_fns)
        ]
        
        # Buffers for pipeline parallelism
        self.activation_buffers = [
            [None for _ in range(num_microbatches + 1)]
            for _ in range(self.num_stages + 1)
        ]
        
    def train_batch(self, full_batch, loss_fn):
        """Train using pipeline parallelism with gradient accumulation."""
        # Split batch into microbatches
        microbatches = self._split_batch(full_batch, self.num_microbatches)
        
        # Zero gradients
        for opt in self.optimizers:
            opt.zero_grad()
            
        # Pipeline forward and backward passes
        for step in range(2 * self.num_stages + self.num_microbatches - 2):
            # Determine which microbatches and stages to process
            for stage in range(self.num_stages):
                microbatch_idx = step - stage
                
                if 0 <= microbatch_idx < self.num_microbatches:
                    # Forward pass for this stage and microbatch
                    if stage == 0:
                        # First stage takes input from the microbatch
                        inputs = microbatches[microbatch_idx][0].to(
                            next(self.partitioned_model[0].parameters()).device
                        )
                    else:
                        # Other stages take input from previous stage
                        inputs = self.activation_buffers[stage][microbatch_idx]
                        
                    # Process through model partition
                    outputs = self.partitioned_model[stage](inputs)
                    
                    # Store activation for next stage
                    if stage < self.num_stages - 1:
                        self.activation_buffers[stage + 1][microbatch_idx] = outputs
                    else:
                        # For last stage, compute loss and backward
                        targets = microbatches[microbatch_idx][1].to(outputs.device)
                        loss = loss_fn(outputs, targets) / self.num_microbatches
                        loss.backward()
        
        # Apply accumulated gradients
        for opt in self.optimizers:
            opt.step()
```

## Sharded Dataset Management

For handling datasets too large to fit in memory:

```python
class ShardedDatasetManager:
    def __init__(self, data_root, shard_size_mb=1024, cache_size_mb=4096):
        self.data_root = data_root
        self.shard_size_mb = shard_size_mb
        self.cache = LRUCache(max_size_mb=cache_size_mb)
        self.shard_index = self._build_shard_index()
        
    def _build_shard_index(self):
        """Build an index of all data shards."""
        index = {}
        for shard_path in glob.glob(os.path.join(self.data_root, "shard_*.index")):
            shard_id = int(os.path.basename(shard_path).split("_")[1].split(".")[0])
            with open(shard_path, 'r') as f:
                shard_contents = json.load(f)
                for item_id, offset in shard_contents.items():
                    index[item_id] = (shard_id, offset)
        return index
        
    def get_item(self, item_id):
        """Retrieve an item by ID, handling shard loading as needed."""
        # Check if item is in cache
        if item_id in self.cache:
            return self.cache[item_id]
            
        # Look up shard and offset
        if item_id not in self.shard_index:
            raise KeyError(f"Item {item_id} not found in dataset")
            
        shard_id, offset = self.shard_index[item_id]
        
        # Load shard if not in memory
        shard_path = os.path.join(self.data_root, f"shard_{shard_id}.data")
        with open(shard_path, 'rb') as f:
            f.seek(offset)
            # Read item header to get size
            item_size = struct.unpack("Q", f.read(8))[0]
            # Read item data
            item_data = f.read(item_size)
            
        # Deserialize item
        item = pickle.loads(item_data)
        
        # Cache for future use
        self.cache[item_id] = item
        
        return item
    
    def get_batch(self, item_ids):
        """Efficiently retrieve a batch of items, minimizing shard loads."""
        # Group by shard to reduce shard loading
        items_by_shard = defaultdict(list)
        for item_id in item_ids:
            if item_id in self.shard_index:
                shard_id, _ = self.shard_index[item_id]
                items_by_shard[shard_id].append(item_id)
        
        # Load items, one shard at a time
        batch = []
        for shard_id, shard_item_ids in items_by_shard.items():
            for item_id in shard_item_ids:
                batch.append(self.get_item(item_id))
                
        return batch
```

## Efficient Circuit Analysis

Scaling our circuit extraction and analysis pipeline:

### Distributed Activation Collection

```python
class DistributedActivationCollector:
    def __init__(self, model, layer_names, world_size, rank):
        self.model = model
        self.layer_names = layer_names
        self.world_size = world_size
        self.rank = rank
        self.hooks = self._register_hooks()
        self.activations = {name: [] for name in layer_names}
        
    def _register_hooks(self):
        """Register forward hooks to collect activations."""
        hooks = []
        for name, layer in self._get_named_layers():
            if name in self.layer_names:
                hook = layer.register_forward_hook(
                    lambda mod, inp, out, name=name: self._save_activation(name, out)
                )
                hooks.append(hook)
        return hooks
    
    def _save_activation(self, name, activation):
        """Save activation from a layer."""
        # Clone and detach to avoid memory leaks
        self.activations[name].append(activation.clone().detach().cpu())
    
    def collect_activations(self, dataloader, batch_limit=None):
        """Collect activations from the dataloader."""
        # Clear previous activations
        for name in self.layer_names:
            self.activations[name] = []
            
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(dataloader):
                if batch_limit and batch_idx >= batch_limit:
                    break
                    
                # Only process examples assigned to this rank
                if batch_idx % self.world_size == self.rank:
                    self.model(inputs.to(self._get_device()))
        
        # Gather activations from all processes
        all_activations = self._gather_activations()
        
        return all_activations
    
    def _gather_activations(self):
        """Gather activations from all processes."""
        all_activations = {name: [] for name in self.layer_names}
        
        for name in self.layer_names:
            # Convert local activations to tensor
            local_acts = torch.cat(self.activations[name], dim=0)
            
            # All-gather to collect from all processes
            gathered_acts = [torch.zeros_like(local_acts) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered_acts, local_acts)
            
            # Concatenate results
            all_activations[name] = torch.cat(gathered_acts, dim=0)
        
        return all_activations
```

### Parallel Dictionary Learning

```python
class ParallelDictionaryLearner:
    def __init__(self, n_components, sparsity_constraint, world_size, rank):
        self.n_components = n_components
        self.sparsity_constraint = sparsity_constraint
        self.world_size = world_size
        self.rank = rank
        self.dictionary = None
        
    def fit(self, data, n_iterations=100):
        """Fit dictionary learning model in parallel."""
        # Split data according to rank
        local_data = self._split_data(data)
        
        # Initialize dictionary (same on all ranks)
        dictionary_shape = (self.n_components, local_data.shape[1])
        np.random.seed(42)  # Ensure same initialization across ranks
        self.dictionary = np.random.randn(*dictionary_shape)
        self.dictionary /= np.linalg.norm(self.dictionary, axis=1, keepdims=True)
        
        for iteration in range(n_iterations):
            # 1. Sparse coding step - compute local coefficients
            local_coeffs = self._sparse_encode(local_data)
            
            # 2. All-reduce to get global coefficients and sufficient statistics
            gram_matrix = local_coeffs.T @ local_coeffs
            covar_matrix = local_data.T @ local_coeffs
            
            # All-reduce statistics
            global_gram = np.zeros_like(gram_matrix)
            global_covar = np.zeros_like(covar_matrix)
            
            # MPI all-reduce (simplified here)
            # torch.distributed.all_reduce(torch.tensor(gram_matrix), op=torch.distributed.ReduceOp.SUM)
            # torch.distributed.all_reduce(torch.tensor(covar_matrix), op=torch.distributed.ReduceOp.SUM)
            
            # 3. Dictionary update step (same on all processes)
            for k in range(self.n_components):
                if global_gram[k, k] > 0:
                    # Update dictionary element
                    self.dictionary[k] = global_covar[:, k] / global_gram[k, k]
                    # Normalize
                    self.dictionary[k] /= max(np.linalg.norm(self.dictionary[k]), 1e-10)
        
        return self.dictionary
    
    def _split_data(self, data):
        """Split data according to rank."""
        n_samples = data.shape[0]
        samples_per_rank = n_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else n_samples
        return data[start_idx:end_idx]
    
    def _sparse_encode(self, data):
        """Compute sparse codes for data using current dictionary."""
        # Implement sparse coding algorithm (e.g., LASSO or OMP)
        # Returns coefficients matrix of shape (n_samples, n_components)
        # Simplified implementation:
        coeffs = np.zeros((data.shape[0], self.n_components))
        for i in range(data.shape[0]):
            coeffs[i] = self._encode_sample(data[i])
        return coeffs
    
    def _encode_sample(self, sample):
        """Encode a single sample using orthogonal matching pursuit."""
        # Simplified OMP implementation
        residual = sample.copy()
        coeffs = np.zeros(self.n_components)
        selected_indices = []
        
        for _ in range(self.sparsity_constraint):
            # Compute correlations with dictionary elements
            correlations = np.abs(self.dictionary @ residual)
            
            # Select most correlated element not yet selected
            valid_indices = [i for i in range(self.n_components) if i not in selected_indices]
            if not valid_indices:
                break
                
            best_idx = valid_indices[np.argmax(correlations[valid_indices])]
            selected_indices.append(best_idx)
            
            # Least squares solution for selected dictionary elements
            D_selected = self.dictionary[selected_indices]
            coeffs_selected = np.linalg.lstsq(D_selected.T, sample, rcond=None)[0]
            
            # Update coefficients and residual
            for j, idx in enumerate(selected_indices):
                coeffs[idx] = coeffs_selected[j]
            
            residual = sample - self.dictionary.T @ coeffs
            
        return coeffs
```

## Memory-Efficient Circuit DB

For managing large collections of neural circuits:

```python
class ScalableCircuitDatabase:
    def __init__(self, db_path, cache_size=100):
        self.db_path = db_path
        self.metadata_db = SqliteDict(f"{db_path}/metadata.sqlite", autocommit=True)
        self.circuit_storage = self._init_storage()
        self.circuit_cache = LRUCache(max_size=cache_size)
        
    def _init_storage(self):
        """Initialize the circuit storage backend."""
        storage_path = f"{self.db_path}/circuits"
        os.makedirs(storage_path, exist_ok=True)
        return storage_path
        
    def add_circuit(self, circuit_id, circuit, metadata):
        """Add a circuit and its metadata to the database."""
        # Store metadata
        self.metadata_db[circuit_id] = metadata
        
        # Serialize and store circuit
        circuit_path = f"{self.circuit_storage}/{circuit_id}.pkl"
        with open(circuit_path, 'wb') as f:
            pickle.dump(circuit, f)
        
        # Add to cache
        self.circuit_cache[circuit_id] = circuit
        
    def get_circuit(self, circuit_id):
        """Retrieve a circuit by ID."""
        # Check cache first
        if circuit_id in self.circuit_cache:
            return self.circuit_cache[circuit_id]
            
        # Load from storage
        circuit_path = f"{self.circuit_storage}/{circuit_id}.pkl"
        if not os.path.exists(circuit_path):
            raise KeyError(f"Circuit {circuit_id} not found")
            
        with open(circuit_path, 'rb') as f:
            circuit = pickle.load(f)
            
        # Add to cache
        self.circuit_cache[circuit_id] = circuit
        
        return circuit
        
    def search_circuits(self, query_dict, limit=10):
        """Search circuits by metadata."""
        results = []
        
        for circuit_id, metadata in self.metadata_db.items():
            match = True
            for key, value in query_dict.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break
                    
            if match:
                results.append((circuit_id, metadata))
                if len(results) >= limit:
                    break
                    
        return results
        
    def get_similar_circuits(self, circuit_id, metric="embedding", k=5):
        """Find circuits similar to the given one."""
        if circuit_id not in self.metadata_db:
            raise KeyError(f"Circuit {circuit_id} not found")
            
        # Get source circuit's metadata
        source_metadata = self.metadata_db[circuit_id]
        
        if metric == "embedding" and "embedding" in source_metadata:
            source_embedding = np.array(source_metadata["embedding"])
            
            # Compute similarity to all circuits
            similarities = []
            for cid, metadata in self.metadata_db.items():
                if cid != circuit_id and "embedding" in metadata:
                    target_embedding = np.array(metadata["embedding"])
                    similarity = cosine_similarity(source_embedding, target_embedding)
                    similarities.append((cid, similarity, metadata))
                    
            # Return top-k similar circuits
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
            
        else:
            # Fallback to tag-based similarity
            if "tags" not in source_metadata:
                return []
                
            source_tags = set(source_metadata["tags"])
            similarities = []
            
            for cid, metadata in self.metadata_db.items():
                if cid != circuit_id and "tags" in metadata:
                    target_tags = set(metadata["tags"])
                    similarity = len(source_tags.intersection(target_tags)) / len(source_tags.union(target_tags))
                    similarities.append((cid, similarity, metadata))
                    
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:k]
```

## Scalable Inference Engine

For running composed circuits at scale:

```python
class ScalableInferenceEngine:
    def __init__(self, circuit_db, max_concurrent=8):
        self.circuit_db = circuit_db
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
    def run_circuit(self, circuit_id, inputs, batch_size=32):
        """Run inference on a circuit in batches."""
        circuit = self.circuit_db.get_circuit(circuit_id)
        
        # Process in batches
        all_results = []
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_results = circuit(batch)
            all_results.append(batch_results)
            
        return torch.cat(all_results, dim=0)
        
    def run_multiple_circuits(self, circuit_ids, inputs):
        """Run multiple circuits in parallel."""
        future_to_circuit = {
            self.executor.submit(self.run_circuit, cid, inputs): cid
            for cid in circuit_ids
        }
        
        results = {}
        for future in as_completed(future_to_circuit):
            circuit_id = future_to_circuit[future]
            try:
                results[circuit_id] = future.result()
            except Exception as e:
                results[circuit_id] = e
                
        return results
        
    def run_composed_circuit(self, composition_plan, inputs):
        """Run a composed circuit by executing subcircuits efficiently."""
        # Extract circuit references and connections
        circuit_refs = composition_plan.get_circuit_references()
        connections = composition_plan.get_connections()
        
        # Topologically sort circuits to determine execution order
        execution_order = self._topological_sort(circuit_refs, connections)
        
        # Execute in order, caching intermediate results
        intermediate_results = {}
        for circuit_ref in execution_order:
            circuit_inputs = self._get_circuit_inputs(circuit_ref, connections, intermediate_results, inputs)
            circuit_outputs = self.run_circuit(circuit_ref.id, circuit_inputs)
            intermediate_results[circuit_ref.id] = circuit_outputs
            
        # Return final outputs
        return self._get_final_outputs(composition_plan, intermediate_results)
```

## Resource Monitoring and Auto-Scaling

For dynamically adapting to workload requirements:

```python
class ResourceMonitor:
    def __init__(self, target_utilization=0.8, check_interval=60, min_workers=1, max_workers=16):
        self.target_utilization = target_utilization
        self.check_interval = check_interval
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.running = False
        self.monitor_thread = None
        
    def start(self):
        """Start the resource monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop the resource monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            # Check resource utilization
            cpu_util = psutil.cpu_percent(interval=1) / 100.0
            mem_util = psutil.virtual_memory().percent / 100.0
            gpu_util = self._get_gpu_utilization() if torch.cuda.is_available() else 0
            
            # Determine overall utilization
            utilization = max(cpu_util, mem_util, gpu_util)
            
            # Adjust worker count based on utilization
            if utilization > self.target_utilization * 1.1:
                # Increase workers if significantly over target
                new_workers = min(self.current_workers + 1, self.max_workers)
                if new_workers > self.current_workers:
                    self.current_workers = new_workers
                    self._scale_workers(self.current_workers)
                    
            elif utilization < self.target_utilization * 0.7:
                # Decrease workers if significantly under target
                new_workers = max(self.current_workers - 1, self.min_workers)
                if new_workers < self.current_workers:
                    self.current_workers = new_workers
                    self._scale_workers(self.current_workers)
            
            # Sleep before next check
            time.sleep(self.check_interval)
            
    def _get_gpu_utilization(self):
        """Get average GPU utilization across all devices."""
        try:
            return sum(torch.cuda.utilization(i) for i in range(torch.cuda.device_count())) / torch.cuda.device_count() / 100.0
        except:
            return 0.0
            
    def _scale_workers(self, new_count):
        """Scale worker processes to the new count."""
        # Implementation depends on the worker management system
        # Could call Kubernetes API, Docker Swarm, or other orchestration
        logging.info(f"Scaling workers to {new_count}")
```

## Implementation Status

Current status of our scalability infrastructure:

- **Distributed Training**: Implemented (v0.9)
- **Model Parallelism**: Implemented (v0.7)
- **Pipeline Parallelism**: Prototype (v0.5)
- **Sharded Dataset**: Implemented (v0.8)
- **Distributed Activation Collection**: Implemented (v0.8)
- **Parallel Dictionary Learning**: Prototype (v0.6)
- **Scalable Circuit DB**: Implemented (v0.7)
- **Scalable Inference**: Prototype (v0.5)
- **Auto-scaling**: Design phase (v0.2)

## Future Directions

Our roadmap for scalability improvements:

1. **Hybrid Parallelism**: Combining data, model, and pipeline parallelism
2. **Sparse Activation Tracking**: Tracking only significant activations to reduce memory usage
3. **Federated Circuit Learning**: Distributed circuit extraction across organizations
4. **Serverless Circuit Execution**: On-demand scaling of circuit inference
5. **Dynamic Circuit Specialization**: Adapting circuit implementations to hardware capabilities 