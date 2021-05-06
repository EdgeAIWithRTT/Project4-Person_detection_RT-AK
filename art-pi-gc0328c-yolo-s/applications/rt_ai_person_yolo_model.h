#ifndef __RT_AI_PERSON_YOLO_MODEL_H
#define __RT_AI_PERSON_YOLO_MODEL_H

/* model info ... */

// model name
#define RT_AI_PERSON_YOLO_MODEL_NAME			"person_yolo"

#define RT_AI_PERSON_YOLO_WORK_BUFFER_BYTES		(53116)

#define AI_PERSON_YOLO_DATA_WEIGHTS_SIZE		(73064)

#define RT_AI_PERSON_YOLO_BUFFER_ALIGNMENT		(4)

#define RT_AI_PERSON_YOLO_IN_NUM				(1)

#define RT_AI_PERSON_YOLO_IN_SIZE_BYTES	{	\
	((160 * 160 * 1) * 4),	\
}
#define RT_AI_PERSON_YOLO_IN_1_SIZE			(160 * 160 * 1)
#define RT_AI_PERSON_YOLO_IN_1_SIZE_BYTES		((160 * 160 * 1) * 4)
#define RT_AI_PERSON_YOLO_IN_TOTAL_SIZE_BYTES		((160 * 160 * 1) * 4)



#define RT_AI_PERSON_YOLO_OUT_NUM				(1)

#define RT_AI_PERSON_YOLO_OUT_SIZE_BYTES	{	\
	((5 * 5 * 30) * 4),	\
}
#define RT_AI_PERSON_YOLO_OUT_1_SIZE			(5 * 5 * 30)
#define RT_AI_PERSON_YOLO_OUT_1_SIZE_BYTES		((5 * 5 * 30) * 4)
#define RT_AI_PERSON_YOLO_OUT_TOTAL_SIZE_BYTES		((5 * 5 * 30) * 4)




#define RT_AI_PERSON_YOLO_TOTAL_BUFFER_SIZE		//unused

#endif	//end
