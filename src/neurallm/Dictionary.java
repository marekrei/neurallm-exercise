package neurallm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 * A helper class for mapping strings to integers, and keeping their counts
 *
 */
public class Dictionary {
	public Map<String,Integer> ids;
	public Map<Integer,Double> counts;
	
	public Dictionary(){
		ids = new LinkedHashMap<String,Integer>();
		counts = new LinkedHashMap<Integer,Double>();
	}
	
	public int add(String token){
		Integer id = ids.get(token);
		if(id == null){
			id = ids.size();
			ids.put(token, id);
			counts.put(id, 1.0);
		}
		else{
			counts.put(id, counts.get(id) + 1.0);
		}
		return id;
	}
	
	public int size(){
		return this.ids.size();
	}
	
	public int getId(String token){
		Integer id = this.ids.get(token);
		if(id == null)
			return -1;
		return id;
	}
	
	public double getCount(String token){
		int id = this.getId(token);
		if(id == -1)
			return 0;
		return this.counts.get(id);
	}
	
	public String getString(Integer id){
		for(Entry<String,Integer> e : this.ids.entrySet())
			if(e.getValue().equals(id))
				return e.getKey();
		return null;
	}
}
