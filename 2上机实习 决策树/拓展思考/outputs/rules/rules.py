def findDecision(obj): #obj[0]: SO2, obj[1]: NO, obj[2]: NO2, obj[3]: NOx, obj[4]: PM10, obj[5]: PM2-5
   # {"feature": "NOx", "instances": 320, "metric_value": 1.9401, "depth": 1}
   if obj[3]<=0.38654680022631427:
      # {"feature": "NO", "instances": 298, "metric_value": 1.6053, "depth": 2}
      if obj[1]<=0.14152198765369065:
         # {"feature": "NO2", "instances": 290, "metric_value": 1.5177, "depth": 3}
         if obj[2]<=0.19830802870767666:
            # {"feature": "SO2", "instances": 288, "metric_value": 1.4884, "depth": 4}
            if obj[0]<=0.20838914450432588:
               # {"feature": "PM10", "instances": 268, "metric_value": 1.4491, "depth": 5}
               if obj[4]<=0.30884519869258353:
                  return 'II'
               elif obj[4]>0.30884519869258353:
                  return 'III'
               else:
                  return 'II'
            elif obj[0]>0.20838914450432588:
               return 'III'
            else:
               return 'II'
         elif obj[2]>0.19830802870767666:
            # {"feature": "SO2", "instances": 2, "metric_value": 1.0, "depth": 4}
            if obj[0]<=0.058:
               return 'III'
            elif obj[0]>0.058:
               return 'IV'
            else:
               return 'III'
         else:
            return 'II'
      elif obj[1]>0.14152198765369065:
         # {"feature": "SO2", "instances": 8, "metric_value": 0.9544, "depth": 3}
         if obj[0]>0.04:
            # {"feature": "PM10", "instances": 6, "metric_value": 0.65, "depth": 4}
            if obj[4]>0.181:
               return 'IV'
            elif obj[4]<=0.181:
               # {"feature": "PM2-5", "instances": 2, "metric_value": 1.0, "depth": 5}
               if obj[5]>0.08:
                  return 'III'
               elif obj[5]<=0.08:
                  return 'IV'
               else:
                  return 'III'
            else:
               return 'IV'
         elif obj[0]<=0.04:
            return 'III'
         else:
            return 'IV'
      else:
         return 'II'
   elif obj[3]>0.38654680022631427:
      # {"feature": "NO", "instances": 22, "metric_value": 1.7168, "depth": 2}
      if obj[1]<=0.268:
         # {"feature": "PM2-5", "instances": 21, "metric_value": 1.519, "depth": 3}
         if obj[5]>0.092:
            # {"feature": "NO2", "instances": 17, "metric_value": 1.3328, "depth": 4}
            if obj[2]<=0.184:
               # {"feature": "SO2", "instances": 12, "metric_value": 0.9799, "depth": 5}
               if obj[0]<=0.111:
                  return 'V'
               elif obj[0]>0.111:
                  return 'VI'
               else:
                  return 'V'
            elif obj[2]>0.184:
               # {"feature": "SO2", "instances": 5, "metric_value": 0.971, "depth": 5}
               if obj[0]>0.043:
                  return 'V'
               elif obj[0]<=0.043:
                  return 'IV'
               else:
                  return 'V'
            else:
               return 'V'
         elif obj[5]<=0.092:
            # {"feature": "SO2", "instances": 4, "metric_value": 0.8113, "depth": 4}
            if obj[0]>0.04:
               return 'IV'
            elif obj[0]<=0.04:
               return 'VI'
            else:
               return 'IV'
         else:
            return 'V'
      elif obj[1]>0.268:
         return 'VII'
      else:
         return 'V'
   else:
      return 'II'
