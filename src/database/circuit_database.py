import sqlite3
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
import datetime

class CircuitDatabase:
    """
    Database for storing extracted neural circuits
    Uses SQLite for storage and provides methods for querying and storing circuit data
    """
    
    def __init__(self, db_path: str = "circuits.db"):
        """
        Initialize the database
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create circuits table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS circuits (
            circuit_id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            task_description TEXT,
            model_architecture TEXT NOT NULL,
            training_details TEXT,
            circuit_structure TEXT,
            interpretation TEXT,
            interface_definition TEXT,
            metadata TEXT,
            creation_date TEXT,
            fidelity REAL,
            extraction_method TEXT
        )
        ''')
        
        # Create tags table for categorizing circuits
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')
        
        # Create circuit_tags join table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS circuit_tags (
            circuit_id TEXT,
            tag_id INTEGER,
            PRIMARY KEY (circuit_id, tag_id),
            FOREIGN KEY (circuit_id) REFERENCES circuits(circuit_id),
            FOREIGN KEY (tag_id) REFERENCES tags(tag_id)
        )
        ''')
        
        # Create activation_examples table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS activation_examples (
            example_id INTEGER PRIMARY KEY AUTOINCREMENT,
            circuit_id TEXT NOT NULL,
            input_data BLOB,
            activation_value REAL,
            description TEXT,
            FOREIGN KEY (circuit_id) REFERENCES circuits(circuit_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_circuit(self,
                   circuit_id: str,
                   task_name: str,
                   model_architecture: Dict[str, Any],
                   circuit_structure: Dict[str, Any],
                   interface_definition: Dict[str, Any],
                   task_description: Optional[str] = None,
                   training_details: Optional[Dict[str, Any]] = None,
                   interpretation: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   fidelity: Optional[float] = None,
                   extraction_method: Optional[str] = None,
                   tags: Optional[List[str]] = None) -> bool:
        """
        Add a new circuit to the database
        
        Args:
            circuit_id: Unique identifier for the circuit
            task_name: Name of the task the circuit was trained on
            model_architecture: Details of the model architecture
            circuit_structure: Representation of the extracted circuit
            interface_definition: Input/output interface specification
            task_description: Human-readable description of the task
            training_details: Details about the training process
            interpretation: Human-readable interpretation of the circuit
            metadata: Additional metadata
            fidelity: Measured fidelity/accuracy of the isolated circuit
            extraction_method: Method used to extract the circuit
            tags: List of tags to associate with the circuit
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if circuit already exists
            cursor.execute("SELECT circuit_id FROM circuits WHERE circuit_id = ?", (circuit_id,))
            if cursor.fetchone():
                print(f"Circuit with ID {circuit_id} already exists")
                conn.close()
                return False
            
            # Convert dictionaries to JSON strings
            model_architecture_json = json.dumps(model_architecture)
            circuit_structure_json = json.dumps(circuit_structure)
            interface_definition_json = json.dumps(interface_definition)
            training_details_json = json.dumps(training_details) if training_details else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            # Get current date and time
            creation_date = datetime.datetime.now().isoformat()
            
            # Insert the circuit
            cursor.execute('''
            INSERT INTO circuits (
                circuit_id, task_name, task_description, model_architecture,
                training_details, circuit_structure, interpretation,
                interface_definition, metadata, creation_date, fidelity, extraction_method
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                circuit_id, task_name, task_description, model_architecture_json,
                training_details_json, circuit_structure_json, interpretation,
                interface_definition_json, metadata_json, creation_date, fidelity, extraction_method
            ))
            
            # Add tags if provided
            if tags:
                for tag in tags:
                    # Add tag if it doesn't exist
                    cursor.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
                    
                    # Get tag ID
                    cursor.execute("SELECT tag_id FROM tags WHERE name = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    
                    # Link tag to circuit
                    cursor.execute('''
                    INSERT INTO circuit_tags (circuit_id, tag_id) VALUES (?, ?)
                    ''', (circuit_id, tag_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding circuit: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()
    
    def get_circuit(self, circuit_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a circuit by ID
        
        Args:
            circuit_id: ID of the circuit to retrieve
            
        Returns:
            Dictionary containing circuit data, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            SELECT circuit_id, task_name, task_description, model_architecture,
                   training_details, circuit_structure, interpretation,
                   interface_definition, metadata, creation_date, fidelity, extraction_method
            FROM circuits
            WHERE circuit_id = ?
            ''', (circuit_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
                
            # Get tags for this circuit
            cursor.execute('''
            SELECT t.name FROM tags t
            JOIN circuit_tags ct ON t.tag_id = ct.tag_id
            WHERE ct.circuit_id = ?
            ''', (circuit_id,))
            
            tags = [tag[0] for tag in cursor.fetchall()]
            
            # Get activation examples
            cursor.execute('''
            SELECT example_id, input_data, activation_value, description
            FROM activation_examples
            WHERE circuit_id = ?
            ''', (circuit_id,))
            
            activation_examples = []
            for ex_row in cursor.fetchall():
                example_id, input_data_blob, activation_value, description = ex_row
                input_data = pickle.loads(input_data_blob)
                activation_examples.append({
                    "example_id": example_id,
                    "input_data": input_data,
                    "activation_value": activation_value,
                    "description": description
                })
            
            # Parse JSON fields
            circuit_data = {
                "circuit_id": row[0],
                "task_name": row[1],
                "task_description": row[2],
                "model_architecture": json.loads(row[3]),
                "training_details": json.loads(row[4]) if row[4] else None,
                "circuit_structure": json.loads(row[5]),
                "interpretation": row[6],
                "interface_definition": json.loads(row[7]),
                "metadata": json.loads(row[8]) if row[8] else None,
                "creation_date": row[9],
                "fidelity": row[10],
                "extraction_method": row[11],
                "tags": tags,
                "activation_examples": activation_examples
            }
            
            return circuit_data
            
        except Exception as e:
            print(f"Error retrieving circuit: {e}")
            return None
            
        finally:
            conn.close()
    
    def query_circuits(self, 
                      task_name: Optional[str] = None,
                      tags: Optional[List[str]] = None,
                      min_fidelity: Optional[float] = None,
                      extraction_method: Optional[str] = None,
                      keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query circuits based on various criteria
        
        Args:
            task_name: Filter by task name
            tags: Filter by tags (circuits must have ALL specified tags)
            min_fidelity: Minimum fidelity/accuracy threshold
            extraction_method: Filter by extraction method
            keyword: Search task description and interpretation fields
            
        Returns:
            List of matching circuit dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query_parts = ["SELECT c.circuit_id FROM circuits c"]
            params = []
            
            # Join with tags table if needed
            if tags:
                for i, tag in enumerate(tags):
                    tag_alias = f"ct{i}"
                    query_parts.append(f"""
                    JOIN circuit_tags {tag_alias} ON c.circuit_id = {tag_alias}.circuit_id
                    JOIN tags t{i} ON {tag_alias}.tag_id = t{i}.tag_id AND t{i}.name = ?
                    """)
                    params.extend([tag])
            
            # Build WHERE clause
            where_clauses = []
            if task_name:
                where_clauses.append("c.task_name = ?")
                params.append(task_name)
                
            if min_fidelity is not None:
                where_clauses.append("c.fidelity >= ?")
                params.append(min_fidelity)
                
            if extraction_method:
                where_clauses.append("c.extraction_method = ?")
                params.append(extraction_method)
                
            if keyword:
                where_clauses.append("(c.task_description LIKE ? OR c.interpretation LIKE ?)")
                keyword_param = f"%{keyword}%"
                params.extend([keyword_param, keyword_param])
            
            if where_clauses:
                query_parts.append("WHERE " + " AND ".join(where_clauses))
            
            full_query = " ".join(query_parts)
            cursor.execute(full_query, params)
            
            circuit_ids = [row[0] for row in cursor.fetchall()]
            
            # Retrieve full circuit data for each matched ID
            results = []
            for circuit_id in circuit_ids:
                circuit_data = self.get_circuit(circuit_id)
                if circuit_data:
                    results.append(circuit_data)
            
            return results
            
        except Exception as e:
            print(f"Error querying circuits: {e}")
            return []
            
        finally:
            conn.close()
    
    def add_activation_example(self, 
                              circuit_id: str, 
                              input_data: Any,
                              activation_value: float,
                              description: Optional[str] = None) -> bool:
        """
        Add an activation example for a circuit
        
        Args:
            circuit_id: ID of the circuit
            input_data: Input that activates the circuit (will be pickled)
            activation_value: Activation strength
            description: Optional description of the example
            
        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if circuit exists
            cursor.execute("SELECT circuit_id FROM circuits WHERE circuit_id = ?", (circuit_id,))
            if not cursor.fetchone():
                print(f"Circuit with ID {circuit_id} does not exist")
                return False
            
            # Pickle the input data
            input_data_blob = pickle.dumps(input_data)
            
            # Insert the activation example
            cursor.execute('''
            INSERT INTO activation_examples (circuit_id, input_data, activation_value, description)
            VALUES (?, ?, ?, ?)
            ''', (circuit_id, input_data_blob, activation_value, description))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding activation example: {e}")
            conn.rollback()
            return False
            
        finally:
            conn.close()


if __name__ == "__main__":
    # Example usage
    db = CircuitDatabase("test_circuits.db")
    
    # Add a test circuit
    circuit_id = "test_add_mod_7_v1"
    task_name = "add_mod_7"
    model_arch = {
        "type": "transformer",
        "layers": 2,
        "hidden_size": 128,
        "heads": 4,
        "params": 400000
    }
    
    circuit_structure = {
        "nodes": [
            {"id": "embed_1", "type": "embedding", "layer": 0},
            {"id": "attn_1_2", "type": "attention_head", "layer": 1, "head_idx": 2},
            {"id": "ffn_1_1", "type": "ffn", "layer": 1},
            {"id": "output", "type": "output", "layer": 2}
        ],
        "edges": [
            {"from": "embed_1", "to": "attn_1_2", "weight": 0.8},
            {"from": "attn_1_2", "to": "ffn_1_1", "weight": 0.6},
            {"from": "ffn_1_1", "to": "output", "weight": 0.9}
        ]
    }
    
    interface_def = {
        "input_format": "sequence",
        "input_tokens": ["START", "num_a", "SEP", "num_b"],
        "output_tokens": ["result"],
        "value_ranges": {"num_a": "0-6", "num_b": "0-6", "result": "0-6"}
    }
    
    # Add the circuit
    success = db.add_circuit(
        circuit_id=circuit_id,
        task_name=task_name,
        model_architecture=model_arch,
        circuit_structure=circuit_structure,
        interface_definition=interface_def,
        task_description="Addition modulo 7 (a + b) % 7",
        fidelity=0.98,
        extraction_method="dictionary_learning",
        tags=["Core_Number", "Primitive", "Arithmetic"]
    )
    
    print(f"Added circuit: {success}")
    
    # Query the circuit
    results = db.query_circuits(tags=["Arithmetic"])
    print(f"Found {len(results)} matching circuits")
    
    # Get full circuit data
    circuit = db.get_circuit(circuit_id)
    if circuit:
        print(f"Retrieved circuit: {circuit['task_name']}")
        print(f"Interface: {circuit['interface_definition']}")
        print(f"Tags: {circuit['tags']}") 