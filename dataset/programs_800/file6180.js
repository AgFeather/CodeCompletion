dex.json = {};

////
//
// dex.json : This module provides routines dealing with json data.
//
////

/**
 * 
 * Take a slice of an array.
 * 
 * @method dex.array.slice
 * @param
 */
dex.json.toCsv = function(json, header)
{
	var csv;
	var ri, ci;
  var data = [];

	// Keys are provided.
	if (arguments.length == 2)
	{
    if (Array.isArray(json))
	  {
      for ( ri = 0; ri < json.length; ri++)
      {
        var row = [];
        for ( ci = 0; ci < header.length; ci++)
        {
          row.push(json[ri][header[ci]]);
        }
        data.push(row);
      }
	  }
    else
    {
      var row = [];
      for ( ci = 0; ci < header.length; ci++)
      {
        row.push(json[ri][header[ci]]);
      }
      data.push(row);
    }
	  return dex.csv.csv(header, data);
	}
	else
	{
		return dex.json.toCsv(json, dex.json.keys(json));
	}
};

dex.json.keys = function(json)
{
	var keyMap = {};
	var keys = [];
  var ri, key;
  
	if (Array.isArray(json))
	{
		for (ri=0; ri<json.length; ri++)
		{
			for (key in json[ri])
			{
				keyMap[key] = true;
			}
		}
	}
	else
	{
		for (key in json)
		{
			keyMap[key] = true;
		}
	}
	
  for (key in keyMap)
  {
  	keys.push(key);
  }
  
  return keys;
};
