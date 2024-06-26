import { useId } from "react";
import { useAppDispatch, useAppSelector } from "../redux/hooks";
import { EAppOperation, EImageFit } from "../types";
import { FaFileUpload } from "react-icons/fa";
import {
  performCurrentOperation,
  setCleanerArgument,
  setCleanerId,
  setDrawerArgument,
  setDrawerId,
  setImageAddress,
  setImageFit,
  setOcrArgument,
  setOcrId,
  setSelectedOperation,
  setTranslatorArgument,
  setTranslatorId,
} from "../redux/slices/app";
import TileRow from "./TileRow";
import SelectTileRow from "./SelectTileRow";
import ArgsTileColumn from "./ArgsTileColumn";

export default function ImageSettings() {
  const dispatch = useAppDispatch();

  const operation = useAppSelector((store) => store.app.operation);
  const translatorId = useAppSelector((a) => a.app.translatorId);
  const translators = useAppSelector((a) => a.app.translators);
  const ocrId = useAppSelector((a) => a.app.ocrId);
  const ocrs = useAppSelector((a) => a.app.ocrs);
  const drawerId = useAppSelector((a) => a.app.drawerId);
  const drawers = useAppSelector((a) => a.app.drawers);
  const cleanerId = useAppSelector((a) => a.app.cleanerId);
  const cleaners = useAppSelector((a) => a.app.cleaners);
  const imageFit = useAppSelector((a) => a.app.imageFit);
  const ocrArgs = useAppSelector((a) => a.app.ocrArgs);
  const translatorArgs = useAppSelector((a) => a.app.translatorArgs);
  const drawerArgs = useAppSelector((a) => a.app.drawerArgs);
  const cleanerArgs = useAppSelector((a) => a.app.cleanerArgs);

  const inputElementId = useId()

  return (
    <div className="tile">
      <TileRow name="Upload Image" style={{height: 60}}>
        {/* <input
          value={imageToLoad}
          type="text"
          onChange={(e) => setImageToLoad(e.target.value)}
        /> */}
        <input
          id={inputElementId}
          type='file'
          onChange={(e) => {
            const result = e.target.files?.item(0) 
            if(result instanceof Blob){
              dispatch(setImageAddress(URL.createObjectURL(result)))
            }
          }}
          style={{display: 'none'}}
          accept="image/*"
        />
        <div style={{ display: 'flex', flexDirection: 'row', width: '100%', height: '100%', gap: "20px"}}>
        <button
        className="upload"
        onClick={() =>
          document.getElementById(inputElementId)?.click()
        }
        >
        <FaFileUpload/>
        </button>
        {/* <button
        className="upload"
        onClick={() =>
          navigator.clipboard.readText().then((a) => {
            dispatch(setImageAddress(a.trim()))
          })
        }
        >
        <FaPaste/>
        </button> */}
        </div>
        
      </TileRow>

      <SelectTileRow
        name="Image Fit"
        items={[EImageFit.FIT_TO_PAGE, EImageFit.SCROLL]}
        value={imageFit}
        toSelectValue={(a) => `${a}`}
        toOriginalValue={parseInt}
        toLabel={(a) =>
          a === EImageFit.FIT_TO_PAGE ? "Fit To Page" : "Scroll"
        }
        onSelected={(a) => {
          dispatch(setImageFit(a));
        }}
      />
      <SelectTileRow
        name="Operation"
        items={[EAppOperation.CLEANING, EAppOperation.TRANSLATION]}
        value={operation}
        toSelectValue={(a) => `${a}`}
        toOriginalValue={parseInt}
        toLabel={(a) => (a === EAppOperation.CLEANING ? "Clean" : "Translate")}
        onSelected={(a) => {
          dispatch(setSelectedOperation(a));
        }}
      />

      <SelectTileRow
        name="Cleaner"
        items={cleaners.map((a, idx) => idx)}
        value={cleanerId}
        toSelectValue={(a) => `${a}`}
        toOriginalValue={parseInt}
        toLabel={(a) => cleaners[a]?.name ?? "Loading"}
        onSelected={(a) => {
          dispatch(setCleanerId(a));
        }}
      />

      {cleanerArgs.length > 0 && (
        <ArgsTileColumn
          category="Cleaner"
          args={cleanerArgs}
          argsInfo={cleaners[cleanerId].args}
          onArgumentUpdated={(idx, val) =>
            dispatch(setCleanerArgument({ index: idx, value: val }))
          }
        />
      )}

      {operation === EAppOperation.TRANSLATION && (
        <>
          <SelectTileRow
            name="Ocr"
            items={ocrs.map((a, idx) => idx)}
            value={ocrId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => ocrs[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setOcrId(a));
            }}
          />

          {ocrArgs.length > 0 && (
            <ArgsTileColumn
              category="Ocr"
              args={ocrArgs}
              argsInfo={ocrs[ocrId].args}
              onArgumentUpdated={(idx, val) =>
                dispatch(setOcrArgument({ index: idx, value: val }))
              }
            />
          )}

          <SelectTileRow
            name="Translator"
            items={translators.map((a, idx) => idx)}
            value={translatorId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => translators[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setTranslatorId(a));
            }}
          />

          {translatorArgs.length > 0 && (
            <ArgsTileColumn
              category="Translator"
              args={translatorArgs}
              argsInfo={translators[translatorId].args}
              onArgumentUpdated={(idx, val) =>
                dispatch(setTranslatorArgument({ index: idx, value: val }))
              }
            />
          )}

          <SelectTileRow
            name="Drawer"
            items={drawers.map((a, idx) => idx)}
            value={drawerId}
            toSelectValue={(a) => `${a}`}
            toOriginalValue={parseInt}
            toLabel={(a) => drawers[a]?.name ?? "Loading"}
            onSelected={(a) => {
              dispatch(setDrawerId(a));
            }}
          />

          {drawerArgs.length > 0 && (
            <ArgsTileColumn
              category="Drawer"
              args={drawerArgs}
              argsInfo={drawers[drawerId].args}
              onArgumentUpdated={(idx, val) =>
                dispatch(setDrawerArgument({ index: idx, value: val }))
              }
            />
          )}
        </>
      )}
      <div className="tile-row">
        <div className="tile-row-content">
          <button
            onClick={() => {
              dispatch(performCurrentOperation());
            }}
          >
            {operation === EAppOperation.CLEANING ? "Clean" : "Translate"}
          </button>
        </div>
      </div>
    </div>
  );
}
